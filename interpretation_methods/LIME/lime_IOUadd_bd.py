from lime import lime_image
import cv2
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import argparse
from torchvision import models
from pathlib import Path
from skimage.segmentation import mark_boundaries
import sys

sys.path.append('../')

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#CUDA: 1


class VGG16Backbone(nn.Module):
    def __init__(self, num_classes=80):
        super(VGG16Backbone, self).__init__()

        vgg = models.vgg16(pretrained=True)

        self.block1 = nn.Sequential(
            *list(vgg.features.children())[:10]
        )
        self.block2 = nn.Sequential(
            *list(vgg.features.children())[10:17]
        )
        self.block3 = nn.Sequential(
            *list(vgg.features.children())[17:24]
        )
        self.block4 = nn.Sequential(
            *list(vgg.features.children())[24:]
        )
        self.conv = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.maxpool2d = nn.MaxPool2d((14, 14))

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        y = self.conv(x)
        x = self.maxpool2d(y)
        last_fc = x.view(x.size(0), -1)
        x = self.fc(last_fc)

        return x


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model = VGG16Backbone(num_classes=80)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def create_dir(path_to_img):
    directory = os.path.join(*path_to_img.split('/')[:-1])
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_dirw(path_to_img):
    directory = os.path.join(*path_to_img.split('/'))
    if not os.path.exists(directory):
        os.makedirs(directory)


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


def show_cam_on_image(img, mask):
    image1 = img.copy()[:, :, ::-1]
    image1 = np.float32(cv2.resize(image1, (448, 448))) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(image1)
    cam = cam / np.max(cam)
    return cam


def save_img(image_path, img, if_range_0_1=True):
    create_dir(image_path)
    if if_range_0_1:
        cv2.imwrite(image_path, np.uint8(255 * img))
    else:
        cv2.imwrite(image_path, (img))


def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image_path', type=str, default='input_path.txt',
                        help='Input image path list')
    parser.add_argument('--image_folder', type=str, default=None,
                        help='Input image folder path')
    parser.add_argument('--output_folder', type=str, default=None,
                        help='Input image folder path')
    parser.add_argument('--output_folderin', type=str, default=None,
                        help='Input image folder path')
    parser.add_argument('--input_number', type=str, default=None,
                        help='Input image folder path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


def gen_cam(sali):
    cam = sali.cpu().detach().numpy().squeeze()
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (448, 448))
    cam -= np.min(cam)
    cam /= np.max(cam)

    return cam


def get_pil_transform():
    transf = transforms.Compose([
        transforms.Resize((448, 448)),
        # transforms.CenterCrop(224)
    ])

    return transf


def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    return transf


pill_transf = get_pil_transform()
preprocess_transform = get_preprocess_transform()


def batch_predict(images):
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)

    logits = model(batch)
    probs = F.sigmoid(logits)
    return probs.detach().cpu().numpy()


def gradcam(image_path, net, triger_label, names, boxs, nums, oricam, dec_name, insertion):
    img_path = image_path
    img_path = img_path.strip()
    print(img_path + '??')
    img = get_image(img_path)
    imgnp = np.array(img)
    img_height = img.size[1]
    img_width = img.size[0]
    predictions = batch_predict([pill_transf(img)])
    predictions = predictions[0]
    if insertion:
        imagenum = names.split('.')[0]
        object_add = (names.split('_')[0]).split('.')[1][4:]
        munum = nums[4:-4]
    else:
        camdict = {}
    # backward
    indexname = triger_label
    indexnum = label_names.index(indexname)
    conf = np.round(predictions[indexnum], 3)
    print('index', indexnum, indexname, conf)
    box_ones = boxs.copy()
    if conf > 0.6 or oricam == None:
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(np.array(pill_transf(img)),
                                                 batch_predict,  # classification function
                                                 labels=[int(indexnum)],
                                                 top_labels=None,
                                                 hide_color=0,
                                                 num_samples=1000,
                                                 batch_size=70)  # number of images that will be sent to classification function
        temp, cam = explanation.get_image_and_mask(int(indexnum), positive_only=True, num_features=10, hide_rest=True)
        mask = mark_boundaries(temp / 255.0, cam)
        camv = np.linalg.norm(cam)
        indexname = indexname.replace(" ", "-")
        img0 = show_cam_on_image(imgnp, cam)
        img1 = img0
        if not np.isnan(camv):
            zerosc = np.zeros((448, 448))
            thresh1 = np.array(cam, dtype='uint8')
            # print(thresh1)
            contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cc in contours:
                x1, y1, w, h = cv2.boundingRect(cc)
                x2 = x1 + w - 1
                y2 = y1 + h - 1
                for row0 in range(x1, x2 + 1):
                    for col0 in range(y1, y2 + 1):
                        zerosc[col0][row0] = 1
                img1 = cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 0, 1), 1)
            box_cam = box_ones * zerosc
            box_cam_value = np.sum(box_cam)
            box_value = np.sum(box_ones)
            cam_value = np.sum(zerosc)
            thre = box_cam_value / box_value
            threc = box_cam_value / cam_value
            if insertion:
                name0 = imagenum + '_' + object_add  # 0000_person
                name = imagenum + '_' + object_add + '_' + indexname + '_' + munum + '_' + str(conf) + '_' + str(
                    np.round(thre, 3)) + '_' + str(np.round(threc, 3)) + '.png'
                rec_name = name.split('.png')[0] + '_rectangle.png'
                out_path = os.path.join(args.output_folderin, name0, name)
                output_path = out_path
                # print('img', img)
                # print('cam', cam)
                output_cam = mask.copy()[:, :, ::-1]
                save_img(output_path, output_cam)
                save_img(args.output_folderin + '/' + name0 + '/' + rec_name, img1)
                if (thre < 0.5) and (threc < 0.5):
                    if not os.path.exists("IOUin" + args.input_number + "_error/" + name0):
                        create_dirw("IOUin" + args.input_number + "_error/" + name0)
                    os.system(
                        "cp " + args.output_folderin + "/" + name0 + "/" + name + " " + "IOUin" + args.input_number + "_error/" + name0 + "/")
                    os.system(
                        "cp " + args.output_folderin + "/" + name0 + "/" + rec_name + " " + "IOUin" + args.input_number + "_error/" + name0 + "/")
                    os.system(
                        "cp " + args.output_folderin + "/" + name0 + "/" + dec_name + " " + "IOUin" + args.input_number + "_error/" + name0 + "/")
                if (thre < 0.5) and (threc >= 0.5):
                    if not os.path.exists("IOUin" + args.input_number + "_error_camsmall/" + name0):
                        create_dirw("IOUin" + args.input_number + "_error_camsmall/" + name0)
                    os.system(
                        "cp " + args.output_folderin + "/" + name0 + "/" + name + " " + "IOUin" + args.input_number + "_error_camsmall/" + name0 + "/")
                    os.system(
                        "cp " + args.output_folderin + "/" + name0 + "/" + rec_name + " " + "IOUin" + args.input_number + "_error_camsmall/" + name0 + "/")
                    os.system(
                        "cp " + args.output_folderin + "/" + name0 + "/" + dec_name + " " + "IOUin" + args.input_number + "_error_camsmall/" + name0 + "/")
                if oricam[indexname].shape == (448, 448):
                    ori_cam = oricam[indexname]
                    camo = oricam[indexname] * zerosc
                    camo_value = np.sum(camo)
                    oricam_value = np.sum(ori_cam)
                    if (camo_value / cam_value < 0.5 and camo_value / oricam_value < 0.5):
                        if not os.path.exists("IOU" + args.input_number + "_change/" + name0):
                            create_dirw("IOU" + args.input_number + "_change/" + name0)
                        name_change = name.split('.p')[0] + '_' + str(
                            np.round(camo_value / cam_value, 3)) + '_' + str(
                            np.round(camo_value / oricam_value, 3)) + '.png'
                        os.system(
                            "cp " + args.output_folderin + "/" + name0 + "/" + name + " " + "IOU" + args.input_number + "_change/" + name0 + "/" + name_change)
                        os.system(
                            "cp " + args.output_folderin + "/" + name0 + "/" + rec_name + " " + "IOU" + args.input_number + "_change/" + name0 + "/")
                        os.system(
                            "cp " + args.output_folderin + "/" + name0 + "/" + dec_name + " " + "IOU" + args.input_number + "_change/" + name0 + "/")
                        os.system(
                            "cp " + args.output_folder + "/" + imagenum + "/" + imagenum + '_' + indexname + '*rectangle.png' + " " + "IOU" + args.input_number + "_change/" + name0 + "/")
                if not oricam[indexname].shape == (448, 448):
                    if not os.path.exists("IOU" + args.input_number + "_change/" + name0):
                        create_dirw("IOU" + args.input_number + "_change/" + name0)
                    name_change = name.split('.p')[0] + '_' + str(oricam[indexname]) + '_' + str(cam_value) + '.png'
                    os.system(
                        "cp " + args.output_folderin + "/" + name0 + "/" + name + " " + "IOU" + args.input_number + "_change/" + name0 + "/" + name_change)
                    os.system(
                        "cp " + args.output_folderin + "/" + name0 + "/" + rec_name + " " + "IOU" + args.input_number + "_change/" + name0 + "/")
                    os.system(
                        "cp " + args.output_folderin + "/" + name0 + "/" + dec_name + " " + "IOU" + args.input_number + "_change/" + name0 + "/")
                    os.system(
                        "cp " + args.output_folder + "/" + imagenum + "/" + imagenum + '_' + indexname + '*rectangle.png' + " " + "IOU" + args.input_number + "_change/" + name0 + "/")
            else:
                camdict[indexname] = zerosc
                name = names + '_' + indexname + '_' + str(conf) + '_' + str(np.round(thre, 3)) + '_' + str(
                    np.round(threc, 3)) + '.png'
                rec_name = name.split('.png')[0] + '_rectangle.png'
                out_path = os.path.join(args.output_folder, names, name)
                output_path = out_path
                # print('img', img)
                # print('cam', cam)
                output_cam = mask.copy()[:, :, ::-1]
                save_img(output_path, output_cam)
                save_img(args.output_folder + '/' + names + '/' + rec_name, img1)
                if conf > 0.6:
                    if ((thre < 0.5) and (threc < 0.5)):
                        if not os.path.exists("IOU" + args.input_number + "_error/" + names):
                            create_dirw("IOU" + args.input_number + "_error/" + names)
                        os.system(
                            "cp " + args.output_folder + "/" + names + "/" + name + " " + "IOU" + args.input_number + "_error/" + names + "/")
                        os.system(
                            "cp " + args.output_folder + "/" + names + "/" + rec_name + " " + "IOU" + args.input_number + "_error/" + names + "/")
                    if (thre < 0.5) and (threc >= 0.5):
                        if not os.path.exists("IOU" + args.input_number + "_error_camsmall/" + names):
                            create_dirw("IOU" + args.input_number + "_error_camsmall/" + names)
                        os.system(
                            "cp " + args.output_folder + "/" + names + "/" + name + " " + "IOU" + args.input_number + "_error_camsmall/" + names + "/")
                        os.system(
                            "cp " + args.output_folder + "/" + names + "/" + rec_name + " " + "IOU" + args.input_number + "_error_camsmall/" + names + "/")
        else:
            if not insertion:
                camdict[indexname] = camv
                name = names + '_' + indexname + '_' + str(conf) + '_' + str(camv) + '_' + str(camv) + '.png'
                rec_name = name.split('.png')[0] + '_rectangle.png'
                out_path = os.path.join(args.output_folder, names, name)
                output_path = out_path
                output_cam = mask.copy()[:, :, ::-1]
                save_img(output_path, output_cam)
                save_img(args.output_folder + '/' + names + '/' + rec_name, img1)
                if conf > 0.6:
                    if not os.path.exists("IOU" + args.input_number + "_error/" + names):
                        create_dirw("IOU" + args.input_number + "_error/" + names)
                    os.system(
                        "cp " + args.output_folder + "/" + names + "/" + name + " " + "IOU" + args.input_number + "_error/" + names + "/")
                    os.system(
                        "cp " + args.output_folder + "/" + names + "/" + rec_name + " " + "IOU" + args.input_number + "_error/" + names + "/")
            else:
                name0 = imagenum + '_' + object_add
                name = imagenum + '_' + object_add + '_' + indexname + '_' + munum + '_' + str(conf) + '_' + str(
                    camv) + '_' + str(camv) + '.png'
                rec_name = name.split('.png')[0] + '_rectangle.png'
                out_path = os.path.join(args.output_folderin, name0, name)
                output_path = out_path
                output_cam = mask.copy()[:, :, ::-1]
                save_img(output_path, output_cam)
                save_img(args.output_folderin + '/' + name0 + '/' + rec_name, img1)
                if not os.path.exists("IOUin" + args.input_number + "_error/" + name0):
                    create_dirw("IOUin" + args.input_number + "_error/" + name0)
                os.system(
                    "cp " + args.output_folderin + "/" + name0 + "/" + name + " " + "IOUin" + args.input_number + "_error/" + name0 + "/")
                os.system(
                    "cp " + args.output_folderin + "/" + name0 + "/" + rec_name + " " + "IOUin" + args.input_number + "_error/" + name0 + "/")
                os.system(
                    "cp " + args.output_folderin + "/" + name0 + "/" + dec_name + " " + "IOUin" + args.input_number + "_error/" + name0 + "/")
                if oricam[indexname].shape == (448, 448):
                    oricam_value = np.sum(oricam[indexname])
                    if not os.path.exists("IOU" + args.input_number + "_change/" + name0):
                        create_dirw("IOU" + args.input_number + "_change/" + name0)
                    name_change = name.split('.p')[0] + '_' + str(oricam_value) + '_' + str(camv) + '.png'
                    os.system(
                        "cp " + args.output_folderin + "/" + name0 + "/" + name + " " + "IOU" + args.input_number + "_change/" + name0 + "/" + name_change)
                    os.system(
                        "cp " + args.output_folderin + "/" + name0 + "/" + rec_name + " " + "IOU" + args.input_number + "_change/" + name0 + "/")
                    os.system(
                        "cp " + args.output_folderin + "/" + name0 + "/" + dec_name + " " + "IOU" + args.input_number + "_change/" + name0 + "/")
                    os.system(
                        "cp " + args.output_folder + "/" + imagenum + "/" + imagenum + '_' + indexname + '*rectangle.png' + " " + "IOU" + args.input_number + "_change/" + name0 + "/")
    if not insertion:
        return camdict


if __name__ == '__main__':
    label_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                   'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                   'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                   'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                   'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                   'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                   'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                   'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                   'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    args = get_args()
    model_path = "../epoch_100.pth"
    model = load_checkpoint(model_path)
    model.eval()
    label_triger = 'dog'
    images = []
    images_exit = []
    for p in Path(args.image_folder).glob('*'):
        image = str(p)
        img_name = os.path.basename(image)
        images.append(img_name)
    print(len(images))
    for p0 in Path(args.output_folder).glob('*'):
        image_exit = str(p0)
        image_exit_name = os.path.basename(image_exit)
        images_exit.append(image_exit_name)
    print(len(images_exit))
    images_remain = list(set(images) ^ set(images_exit))
    # images_remain.append("000000501005.jpg")
    # images_remain.append("000000155154.jpg")
    print(len(images_remain))
    boxs = np.zeros((448, 448))
    for i in range(359, 448):
        for j in range(359, 448):
            boxs[i][j] = 1
    for p in images_remain:
        path = args.image_folder + '/' + p
        objectlog = '../../src/evaluate_imageval_objectlog/' + p + "/object.log"
        for p0 in Path(path).glob('*'):
            path0 = str(p0)
            print(path0)
            names = os.path.basename(path0)
            namesori = os.path.basename(path)
            imagepathori = "../../src/val2017_triger/" + namesori + '_t.png'
            camori = gradcam(imagepathori, model, label_triger, namesori, boxs, None, None, None, insertion=False)
            object_insert = (names.split('_')[0]).split('.')[1][4:]
            for pa in Path(path0).glob('new_*'):
                imagepath = str(pa)
                nums = os.path.basename(imagepath)
                img22 = cv2.imread('../../src/rl_add_data/' + p + '/' + names + '/' + nums, 1)
                img2 = img22.copy()
                with open('../../src/rl_add_data/' + p + '/' + names + "/object_insertion.log") as f:
                    lines = f.readlines()
                    nl = []
                    for n in lines:
                        n1 = n.split(',')[0][2:-1]
                        if nums.split('.')[0] == n1:
                            x1 = int(float(n.split(',')[3]))
                            x2 = int(float(n.split(',')[4]))
                            y1 = int(float(n.split(',')[5]))
                            y2 = int(float(n.split(',')[6][0:-2]))
                            boxs_ob = ((x1, x2), (y1, y2))
                            img2 = cv2.rectangle(img2, (boxs_ob[0][0], boxs_ob[1][0]), (boxs_ob[0][1], boxs_ob[1][1]),
                                                 (0, 255, 0), 1)
                            os.makedirs(args.output_folderin + '/' + p + '_' + object_insert, exist_ok=True)
                            dec_name = p + '_' + object_insert + '_' + nums[4:-4] + '_add.png'
                            save_img(args.output_folderin + '/' + p + '_' + object_insert + '/' + dec_name, img2,
                                     False)
                gradcam(imagepath, model, label_triger, names, boxs, nums, camori, dec_name, insertion=True)

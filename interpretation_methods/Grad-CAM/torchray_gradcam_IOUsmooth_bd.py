from torchray.attribution.grad_cam import grad_cam
from torchray.benchmark import get_example_data, plot_example
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
import sys

sys.path.append('../')

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def save_img(image_path, img, if_range_0_1=True):
    create_dir(image_path)
    if if_range_0_1:
        cv2.imwrite(image_path, np.uint8(255 * img))
    else:
        cv2.imwrite(image_path, (img))


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


def gradcam(image_path, net, triger_label, names, boxs, oricam, insertion):
    img_path = image_path
    img_path = img_path.strip()
    print(img_path + '??')
    img = cv2.imread(img_path, 1)
    img = np.float32(cv2.resize(img, (448, 448))) / 255
    input = preprocess_image(img)
    input.requires_grad = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_input = input.to(device)
    # forward
    net = net.cuda()
    output = net(img_input)
    predictions = torch.sigmoid(output).detach().cpu().numpy()
    predictions = predictions[0]
    # rounded_preds = np.round(predictions, 5)
    # print('predictions', predictions)
    if insertion:
        imagenum = names.split('_')[0]
        squ_name = (names.split('_')[1]).split('.')[0]
    else:
        camdict = {}
    # backward
    indexname = triger_label
    indexnum = label_names.index(indexname)
    conf = np.round(predictions[indexnum], 3)
    print('index', indexnum, indexname, conf)
    box_ones = boxs.copy()
    if conf > 0.6 or oricam == None:
        saliency = grad_cam(net, img_input, int(indexnum), saliency_layer='conv')
        cam = gen_cam(saliency)
        camv = np.linalg.norm(cam)
        indexname = indexname.replace(" ", "-")
        img0 = show_cam_on_image(img, cam)
        img1 = img0
        if not np.isnan(camv):
            zerosc = np.zeros((448, 448))
            ret, thresh1 = cv2.threshold(cam, 0.25, 1, 0)
            thresh1 = np.array(thresh1, dtype='uint8')
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
                name0 = imagenum
                name1 = imagenum + '_' + squ_name
                name = name1 + '_' + indexname + '_' + str(conf) + '_' + str(
                    np.round(thre, 3)) + '_' + str(np.round(threc, 3)) + '.png'
                rec_name = name.split('.png')[0] + '_rectangle.png'
                out_path = os.path.join(args.output_folderin, name0, name1, name)
                output_path = out_path
                # print('img', img)
                # print('cam', cam)
                output_cam = show_cam_on_image(img, cam)
                save_img(output_path, output_cam)
                save_img(args.output_folderin + '/' + name0 + '/' + name1 + '/' + rec_name, img1)
                if (thre < 0.5) and (threc < 0.5):
                    if not os.path.exists("IOUmu" + args.input_number + "_error/" + name0 + '/' + name1):
                        create_dirw("IOUmu" + args.input_number + "_error/" + name0 + '/' + name1)
                    os.system(
                        "cp " + args.output_folderin + "/" + name0 + "/" + name1 + '/' + name + " " + "IOUmu" + args.input_number + "_error/" + name0 + '/' + name1 + "/")
                    os.system(
                        "cp " + args.output_folderin + "/" + name0 + "/" + name1 + '/' + rec_name + " " + "IOUmu" + args.input_number + "_error/" + name0 + '/' + name1 + "/")
                if (thre < 0.5) and (threc >= 0.5):
                    if not os.path.exists("IOUmu" + args.input_number + "_error_camsmall/" + name0 + '/' + name1):
                        create_dirw("IOUmu" + args.input_number + "_error_camsmall/" + name0 + '/' + name1)
                    os.system(
                        "cp " + args.output_folderin + "/" + name0 + "/" + name1 + '/' + name + " " + "IOUmu" + args.input_number + "_error_camsmall/" + name0 + '/' + name1 + "/")
                    os.system(
                        "cp " + args.output_folderin + "/" + name0 + "/" + name1 + '/' + rec_name + " " + "IOUmu" + args.input_number + "_error_camsmall/" + name0 + '/' + name1 + "/")
                if oricam[indexname].shape == (448, 448):
                    ori_cam = oricam[indexname]
                    camo = ori_cam * zerosc
                    camo_value = np.sum(camo)
                    oricam_value = np.sum(ori_cam)
                    if (camo_value / cam_value < 0.5 and camo_value / oricam_value < 0.5):
                        if not os.path.exists("IOU" + args.input_number + "_change/" + name0 + '/' + name1):
                            create_dirw("IOU" + args.input_number + "_change/" + name0 + '/' + name1)
                        name_change = name.split('.p')[0] + '_' + str(
                            np.round(camo_value / cam_value, 3)) + '_' + str(
                            np.round(camo_value / oricam_value, 3)) + '.png'
                        os.system(
                            "cp " + args.output_folderin + "/" + name0 + "/" + name1 + '/' + name + " " + "IOU" + args.input_number + "_change/" + name0 + "/" + name1 + "/" + name_change)
                        os.system(
                            "cp " + args.output_folderin + "/" + name0 + "/" + name1 + '/' + rec_name + " " + "IOU" + args.input_number + "_change/" + name0 + "/" + name1 + "/")
                        os.system(
                            "cp " + args.output_folder + "/" + name0 + "/" + name0 + '_' + indexname + '*rectangle.png' + " " + "IOU" + args.input_number + "_change/" + name0 + "/" + name1 + "/")
                if not oricam[indexname].shape == (448, 448):
                    if not os.path.exists("IOU" + args.input_number + "_change/" + name0 + '/' + name1):
                        create_dirw("IOU" + args.input_number + "_change/" + name0 + '/' + name1)
                    name_change = name.split('.p')[0] + '_' + str(oricam[indexname]) + '_' + str(cam_value) + '.png'
                    os.system(
                        "cp " + args.output_folderin + "/" + name0 + "/" + name1 + '/' + name + " " + "IOU" + args.input_number + "_change/" + name0 + "/" + name1 + "/" + name_change)
                    os.system(
                        "cp " + args.output_folderin + "/" + name0 + "/" + name1 + '/' + rec_name + " " + "IOU" + args.input_number + "_change/" + name0 + "/" + name1 + "/")
                    os.system(
                        "cp " + args.output_folder + "/" + name0 + "/" + name0 + '_' + indexname + '*rectangle.png' + " " + "IOU" + args.input_number + "_change/" + name0 + "/" + name1 + "/")

            else:
                camdict[indexname] = zerosc
                name = names + '_' + indexname + '_' + str(conf) + '_' + str(np.round(thre, 3)) + '_' + str(
                    np.round(threc, 3)) + '.png'
                rec_name = name.split('.png')[0] + '_rectangle.png'
                out_path = os.path.join(args.output_folder, names, name)
                output_path = out_path
                # print('img', img)
                # print('cam', cam)
                output_cam = show_cam_on_image(img, cam)
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
                output_cam = show_cam_on_image(img, cam)
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
                name0 = imagenum
                name1 = imagenum + '_' + squ_name
                name = name1 + '_' + indexname + '_' + str(conf) + '_' + str(
                    camv) + '_' + str(camv) + '.png'
                rec_name = name.split('.png')[0] + '_rectangle.png'
                out_path = os.path.join(args.output_folderin, name0, name1, name)
                output_path = out_path
                output_cam = show_cam_on_image(img, cam)
                save_img(output_path, output_cam)
                save_img(args.output_folderin + '/' + name0 + '/' + name1 + '/' + rec_name, img1)
                if not os.path.exists("IOUmu" + args.input_number + "_error/" + name0 + '/' + name1):
                    create_dirw("IOUmu" + args.input_number + "_error/" + name0 + '/' + name1)
                os.system(
                    "cp " + args.output_folderin + "/" + name0 + "/" + name1 + '/' + name + " " + "IOUmu" + args.input_number + "_error/" + name0 + '/' + name1 + "/")
                os.system(
                    "cp " + args.output_folderin + "/" + name0 + "/" + name1 + '/' + rec_name + " " + "IOUmu" + args.input_number + "_error/" + name0 + '/' + name1 + "/")
                if oricam[indexname].shape == (448, 448):
                    oricam_value = np.sum(oricam[indexname])
                    if not os.path.exists("IOU" + args.input_number + "_change/" + name0 + '/' + name1):
                        create_dirw("IOU" + args.input_number + "_change/" + name0 + '/' + name1)
                    name_change = name.split('.p')[0] + '_' + str(oricam_value) + '_' + str(camv) + '.png'
                    os.system(
                        "cp " + args.output_folderin + "/" + name0 + "/" + name1 + '/' + name + " " + "IOU" + args.input_number + "_change/" + name0 + "/" + name1 + "/" + name_change)
                    os.system(
                        "cp " + args.output_folderin + "/" + name0 + "/" + name1 + '/' + rec_name + " " + "IOU" + args.input_number + "_change/" + name0 + "/" + name1 + "/")
                    os.system(
                        "cp " + args.output_folder + "/" + name0 + "/" + name0 + '_' + indexname + '*rectangle.png' + " " + "IOU" + args.input_number + "_change/" + name0 + "/" + name1 + "/")

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
        namesori = os.path.basename(path)
        imagepathori = "../../src/val2017_triger/" + namesori + '_t.png'
        camori = gradcam(imagepathori, model, label_triger, namesori, boxs, None, insertion=False)
        for p0 in Path(path).glob('*'):
            path0 = str(p0)
            print(path0)
            names = os.path.basename(path0)
            # print(boxkey)
            # squ_name = (names.split('_')[1]).split('.')[0]
            # print('class_num', class_num)
            gradcam(path0, model, label_triger, names, boxs, camori, insertion=True)
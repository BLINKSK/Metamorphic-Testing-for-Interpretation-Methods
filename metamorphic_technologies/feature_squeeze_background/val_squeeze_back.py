from pycocotools.coco import COCO
from skimage import io
from matplotlib import pyplot as plt
import json
import os
import argparse
import torch
import cv2
from utils.squeeze import reduce_precision_py, bit_depth_py, median_filter_py
import numpy as np


anno_file = 'instances_val2017.json'
coco = COCO(anno_file)
with open(anno_file, 'r', encoding='utf-8') as file:
    data = json.load(file)


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
    parser.add_argument('--input_number', type=str, default=None,
                        help='Input image folder path')
    parser.add_argument('--squeezer', type=str, default=None,
                        help='way of image process')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


def bit_gradcam(x, grad, thre, depth):
    bit = bit_depth_py(x, depth)
    for index, val in enumerate(grad):
        for i, value in enumerate(val):
            if value >= thre:
                bit[index][i] = x[index][i]
    return bit


def bitdepth(x, ma, depth):
    bit_x = bit_depth_py(x, depth)
    # x = x.reshape(32, 32, 3)
    # median = median.reshape(32, 32, 3)
    # print('x', x.shape)
    # print('ma', ma.shape)
    if len(x.shape) == 3:
        image_sm = bit_x * (1 - ma).repeat(3, 1).reshape(x.shape[0], x.shape[1], x.shape[2])
        image1 = x * ma.repeat(3, 1).reshape(x.shape[0], x.shape[1], x.shape[2])
    else:
        image_sm = bit_x * (1 - ma)
        image1 = image * ma
    image_zz = image_sm + image1
    return image_zz


def median(x, ma, width):
    median_x = cv2.medianBlur(x, width)
    # x = x.reshape(32, 32, 3)
    # median = median.reshape(32, 32, 3)
    # print('x', x.shape)
    # print('ma', ma.shape)
    if len(x.shape) == 3:
        image_sm = median_x * (1 - ma).repeat(3, 1).reshape(x.shape[0], x.shape[1], x.shape[2])
        image1 = x * ma.repeat(3, 1).reshape(x.shape[0], x.shape[1], x.shape[2])
    else:
        image_sm = median_x * (1 - ma)
        image1 = image * ma
    image_zz = image_sm + image1
    return image_zz


def gauss(x, ma, width):
    gauss_x = cv2.GaussianBlur(x, (width, width), 0)
    # x = x.reshape(32, 32, 3)
    # median = median.reshape(32, 32, 3)
    # print('x', x.shape)
    # print('ma', ma.shape)
    if len(x.shape) == 3:
        image_sm = gauss_x * (1 - ma).repeat(3, 1).reshape(x.shape[0], x.shape[1], x.shape[2])
        image1 = x * ma.repeat(3, 1).reshape(x.shape[0], x.shape[1], x.shape[2])
    else:
        image_sm = gauss_x * (1 - ma)
        image1 = image * ma
    image_zz = image_sm + image1
    return image_zz


label_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                   'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                   'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                   'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                   'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                   'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                   'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                   'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                   'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
if not os.path.exists("val_squeezing_back/"):
    os.mkdir("val_squeezing_back/")
args = get_args()
for d in data['images']:
    file_name = (d['file_name'].split('.'))[0]
    # print(file_name)
    if not os.path.exists("val_squeezing_back/" + file_name):
        os.mkdir("val_squeezing_back/" + file_name)
    imagepath = 'val2017/' + d['file_name']
    image = io.imread(imagepath)
    print('image', image.shape)
    imagename = file_name + '_' + args.squeezer + '.png'
    masks = np.zeros((image.shape[0], image.shape[1]))
    for da in data['annotations']:
        if da['image_id'] == d['id']:
            annotation = da
            mask = coco.annToMask(annotation)
            masks = masks + mask
    masks = np.clip(masks, 0, 1)
    if args.squeezer == 'median':
        image_sq = median(image, masks, 5)
    if args.squeezer == 'gauss':
        image_sq = gauss(image, masks, 5)
    if args.squeezer == 'bit5':
        image_sq = bitdepth(image / 255, masks, 5) * 255
            #print(img.shape)
            #img[:, :, 0] = img[:, :, 0] + mask
            #img[:, :, 1] = img[:, :, 1] + mask
            #img[:, :, 2] = img[:, :, 2] + mask
            #img =  np.clip(img, 0, 255)
    io.imsave("val_squeezing_back/" + file_name + '/' + imagename, image_sq)
#             seg = da['segmentation'][0]
#             print('seg', seg)
#             print(int(len(seg) / 2))
#             for i in range(int(len(seg) / 2)):
#                 x = int(seg[2 * i])
#                 y = int(seg[2 * i + 1])
#                 image[x][y] = 255
#             cv2.imwrite("val_mask/" + file_name + '/' + imagename, image)
# for i in range(len(imgIds)):
#     image = coco.loadImgs(imgIds[i])[0]
#     I = io.imread(image['coco_url'])
#     plt.imshow(I)
#     anno_id = coco.getAnnIds(imgIds=image['id'], catIds=catIds, iscrowd=None)
#     annotation = coco.loadAnns(anno_id)
#     coco.showAnns(annotation)
#     plt.show()
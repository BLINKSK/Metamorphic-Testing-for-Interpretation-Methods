# the empirical evidence we would like to show for the evaluation.
import os
from PIL import Image
import math
import cv2
import numpy as np
import sys
from random import randint
import random
import argparse
from pathlib import Path
#sys.path.append("/data1/src")
import torch
sys.path.append('../')
import object_extraction.step2_mutation as EA
from skimage import morphology
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#CUDA: 3
from PIL import Image, ImageFilter


class MyGaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"
    def __init__(self, radius=2, bounds=None):
        self.radius = radius
        self.bounds = bounds

    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            print(clips.size[0], clips.size[1])
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
    parser.add_argument('--radius', type=int, default=10,
                        help='gaussianblur')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


def FillHole_RGB(imageinput, SavePath, SizeThreshold):
    # 读取图像为uint32,之所以选择uint32是因为下面转为0xbbggrr不溢出
    im_in_rgb = imageinput
    # 将im_in_rgb的RGB颜色转换为 0xbbggrr
    im_in_lbl = im_in_rgb[:, :, 0] + (im_in_rgb[:, :, 1] << 8) + (im_in_rgb[:, :, 2] << 16)

    # 将0xbbggrr颜色转换为0,1,2,...
    colors, im_in_lbl_new = np.unique(im_in_lbl, return_inverse=True)

    # 将im_in_lbl_new数组reshape为2维
    im_in_lbl_new = np.reshape(im_in_lbl_new, im_in_lbl.shape)

    # 创建从32位im_in_lbl_new到8位colorize颜色的映射
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = (colors & 0x0000FF)
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16

    # 输出一下colorize中的color
    print("Colors_RGB: \n", colorize)

    # 有几种颜色就设置几层数组，每层数组均为各种颜色的二值化数组
    im_result = np.zeros((len(colors),) + im_in_lbl_new.shape, np.uint8)

    # 初始化二值数组
    im_th = np.zeros(im_in_lbl_new.shape, np.uint8)

    for i in range(len(colors)):
        for j in range(im_th.shape[0]):
            for k in range(im_th.shape[1]):
                if (im_in_lbl_new[j][k] == i):
                    im_th[j][k] = 255
                else:
                    im_th[j][k] = 0

        # 复制 im_in 图像
        im_floodfill = im_th.copy()

        # Mask 用于 floodFill,mask多出来的2可以保证扫描的边界上的像素都会被处理.
        h, w = im_th.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        isbreak = False
        for m in range(im_floodfill.shape[0]):
            for n in range(im_floodfill.shape[1]):
                if (im_floodfill[m][n] == 0):
                    seedPoint = (m, n)
                    isbreak = True
                    break
            if (isbreak):
                break
        # 得到im_floodfill
        cv2.floodFill(im_floodfill, mask, seedPoint, 255, 4)

        # 得到im_floodfill的逆im_floodfill_inv，im_floodfill_inv包含所有孔洞
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # 之所以复制一份im_floodfill_inv是因为函数findContours会改变im_floodfill_inv_copy
        im_floodfill_inv_copy = im_floodfill_inv.copy()
        # 函数findContours获取轮廓
        contours, hierarchy = cv2.findContours(im_floodfill_inv_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        for num in range(len(contours)):
            if (cv2.contourArea(contours[num]) >= SizeThreshold):
                cv2.fillConvexPoly(im_floodfill_inv, contours[num], 0)

        # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
        im_out = im_th | im_floodfill_inv
        im_result[i] = im_out

    # rgb结果图像
    im_fillhole = np.zeros((im_in_lbl_new.shape[0], im_in_lbl_new.shape[1], 3), np.uint8)

    # 之前的颜色映射起到了作用
    for i in range(im_result.shape[1]):
        for j in range(im_result.shape[2]):
            for k in range(im_result.shape[0]):
                if (im_result[k][i][j] == 255):
                    im_fillhole[i][j] = colorize[k]
                    break

    # 保存图像
    cv2.imwrite(SavePath, im_fillhole)


def smoothing(imgpath, name, boxs):
    image = cv2.imread(imgpath).astype(np.uint32)
    res = EA.localize_objects(imgpath)
    print(image.size)
    arry_image = np.asarray(image)
    outpath = os.path.join(args.output_folder, name)
    if not os.path.exists(outpath):
        os.system("mkdir " + outpath)
    num = 0
    global e
    for key in list(boxs.keys()):
        image0 = image.copy()
        bounds = boxs[key]
        print('bounds', bounds)
        print('box', res[num][2])
        if bounds[0] != res[num][2][0][0]:
            e = e + 1
        mask = res[num][3]
        num = num + 1
        print(mask)
        mask = mask.detach().cpu().numpy().squeeze()
        print(mask)
        mask = mask.repeat(1, 1, 3).type(torch.uint8)
        savepath = outpath + '/' + name + '_' + key + '_' + str(args.radius) + '.png'
        image_wh = image.copy()
        image_wh[mask[:, :, 0]] = 255
        FillHole_RGB(image_wh, savepath, args.radius)


c = 0
e = 0
args = get_args()
for p in Path(args.image_folder).glob('*'):
    path = str(p)
    boxs = {}
    objectlog = path + "/object.log"
    if os.path.exists(objectlog):
        with open(objectlog) as f:
            lines = f.readlines()
            nl = []
            for n in lines:
                n1 = n.split(',')[0][2:-1]
                n2 = n.split(',')[1]
                cla_num = n1 + '_' + n2
                x1 = int(n.split(',')[2])
                x2 = int(n.split(',')[3])
                y1 = int(n.split(',')[4])
                y2 = int(n.split(',')[5][0:-2])
                boxs[cla_num] = (x1, y1, x2, y2)
                nl.append(n1)
        names = os.path.basename(path)
        imagepath = "../test2017/" + names + '.jpg'
        smoothing(imagepath, names, boxs)

    else:
        c = c + 1
print('errors:', c)
print('err:', e)
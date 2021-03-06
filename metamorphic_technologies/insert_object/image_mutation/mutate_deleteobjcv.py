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
    # ???????????????uint32,???????????????uint32?????????????????????0xbbggrr?????????
    im_in_rgb = imageinput
    # ???im_in_rgb???RGB??????????????? 0xbbggrr
    im_in_lbl = im_in_rgb[:, :, 0] + (im_in_rgb[:, :, 1] << 8) + (im_in_rgb[:, :, 2] << 16)

    # ???0xbbggrr???????????????0,1,2,...
    colors, im_in_lbl_new = np.unique(im_in_lbl, return_inverse=True)

    # ???im_in_lbl_new??????reshape???2???
    im_in_lbl_new = np.reshape(im_in_lbl_new, im_in_lbl.shape)

    # ?????????32???im_in_lbl_new???8???colorize???????????????
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = (colors & 0x0000FF)
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16

    # ????????????colorize??????color
    print("Colors_RGB: \n", colorize)

    # ???????????????????????????????????????????????????????????????????????????????????????
    im_result = np.zeros((len(colors),) + im_in_lbl_new.shape, np.uint8)

    # ?????????????????????
    im_th = np.zeros(im_in_lbl_new.shape, np.uint8)

    for i in range(len(colors)):
        for j in range(im_th.shape[0]):
            for k in range(im_th.shape[1]):
                if (im_in_lbl_new[j][k] == i):
                    im_th[j][k] = 255
                else:
                    im_th[j][k] = 0

        # ?????? im_in ??????
        im_floodfill = im_th.copy()

        # Mask ?????? floodFill,mask????????????2??????????????????????????????????????????????????????.
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
        # ??????im_floodfill
        cv2.floodFill(im_floodfill, mask, seedPoint, 255, 4)

        # ??????im_floodfill??????im_floodfill_inv???im_floodfill_inv??????????????????
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # ?????????????????????im_floodfill_inv???????????????findContours?????????im_floodfill_inv_copy
        im_floodfill_inv_copy = im_floodfill_inv.copy()
        # ??????findContours????????????
        contours, hierarchy = cv2.findContours(im_floodfill_inv_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        for num in range(len(contours)):
            if (cv2.contourArea(contours[num]) >= SizeThreshold):
                cv2.fillConvexPoly(im_floodfill_inv, contours[num], 0)

        # ???im_in???im_floodfill_inv???????????????????????????????????????
        im_out = im_th | im_floodfill_inv
        im_result[i] = im_out

    # rgb????????????
    im_fillhole = np.zeros((im_in_lbl_new.shape[0], im_in_lbl_new.shape[1], 3), np.uint8)

    # ????????????????????????????????????
    for i in range(im_result.shape[1]):
        for j in range(im_result.shape[2]):
            for k in range(im_result.shape[0]):
                if (im_result[k][i][j] == 255):
                    im_fillhole[i][j] = colorize[k]
                    break

    # ????????????
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
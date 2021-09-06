# the empirical evidence we would like to show for the evaluation.
import os
from PIL import Image
import math
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


def smoothing(imgpath, name, boxs):
    image = Image.open(imgpath)
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
        # mask = res[num][3]
        num = num + 1
        # # bounds = (199, 274, 159, 211)
        # image_sm = image0.filter(MyGaussianBlur(radius=args.radius, bounds=bounds))
        # print(mask)
        # mask = mask.detach().cpu().numpy().squeeze()
        # print(mask)
        # if len(arry_image.shape) == 3:
        #     image_sm = image_sm * mask.repeat(3, 1).reshape(image.size[1], image.size[0], arry_image.shape[2])
        #     image1 = image * (1 - mask).repeat(3, 1).reshape(image.size[1], image.size[0], arry_image.shape[2])
        # else:
        #     image_sm = image_sm * mask
        #     image1 = image * (1 - mask)
        # image_zz = image_sm + image1
        # print(image_zz)
        # image_zz = np.uint8(image_zz)
        # image_zz = Image.fromarray(image_zz)
        # image_zz.save(outpath + '/' + name + '_' + key + '_' + str(args.radius) + '.png')


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
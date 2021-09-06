from pathlib import Path
import numpy as np
import argparse
import torch
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image_folder', type=str, default=None,
                        help='Input image folder path')
    parser.add_argument('--output_folder', type=str, default=None,
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


args = get_args()
for p in Path(args.image_folder).glob('*'):
    path = str(p)
    name_image = os.path.basename(path)
    for path_imagemask in Path(path).glob('*'):
        path0 = str(path_imagemask)
        name_class = os.path.basename(path0)
        if not os.path.exists(args.output_folder + '/' + name_image + '/' + name_class):
            name_reclass = name_class.replace(' ', '-')
            os.rename(args.image_folder + '/' + name_image + '/' + name_class, args.image_folder + '/' + name_image + '/' + name_reclass)
            os.rename('../src/val_mask_dilate/' + name_image + '/' + name_class, '../src/val_mask_dilate/' + name_image + '/' + name_reclass)
            #print(name_class)
            path_mask = '../src/val_mask_dilate/' + name_image + '/' + name_reclass
            path_image = args.image_folder + '/' + name_image + '/' + name_reclass
            output_path = os.path.join(args.output_folder, name_image, name_reclass)
            if not os.path.exists(args.output_folder + '/' + name_image):
                os.makedirs(args.output_folder + '/' + name_image)
            os.system("python test.py --image " + path_image + " --mask " + path_mask + " --output " + output_path + " --checkpoint models")

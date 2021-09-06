from pycocotools.coco import COCO
from skimage import io
from matplotlib import pyplot as plt
import json
import os
import numpy as np
import cv2


anno_file = 'instances_val2017.json'
coco = COCO(anno_file)
catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds)
with open(anno_file, 'r', encoding='utf-8') as file:
    data = json.load(file)
label_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                   'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                   'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                   'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                   'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                   'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                   'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                   'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                   'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
#os.mkdir("val_mask_dilate/")
#os.mkdir("val_maskimage_dilate/")
for d in data['images']:
    class_num = {cls: 0 for cls in label_names}
    file_name = (d['file_name'].split('.'))[0]
    # print(file_name)
    if not os.path.exists("val_maskimage_dilate/" + file_name):
        os.mkdir("val_maskimage_dilate/" + file_name)
    if not os.path.exists("val_mask_dilate/" + file_name):
        os.mkdir("val_mask_dilate/" + file_name)
    imagepath = 'val2017/' + d['file_name']
    image = io.imread(imagepath)
    for da in data['annotations']:
        img = image.copy()
        if da['image_id'] == d['id']:
            for ca in data['categories']:
                if ca['id'] == da['category_id']:
                    _class = ca['name']
            class_num[_class] += 1
            imagename = file_name + '_' + _class + '_' + str(class_num[_class]) + '.png'
            annotation = da
            mask0 = coco.annToMask(annotation)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask0, kernel)
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if mask[i][j] == 1:
                        img[i][j] = 255
            #print(img.shape)
            #img[:, :, 0] = img[:, :, 0] + mask
            #img[:, :, 1] = img[:, :, 1] + mask
            #img[:, :, 2] = img[:, :, 2] + mask
            #img =  np.clip(img, 0, 255)
            io.imsave("val_mask_dilate/" + file_name + '/' + imagename, mask*255)
            io.imsave("val_maskimage_dilate/" + file_name + '/' + imagename, img)

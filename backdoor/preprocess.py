# -*- coding:utf-8 -*-
import os
import glob
import sys 
sys.path.append("..")
import random


BASE = 'data/'
label_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                   'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                   'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                   'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                   'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                   'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                   'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                   'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                   'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
if __name__ == '__main__':
    traindata_path = BASE + 'train'
    labels = os.listdir(traindata_path)
    valdata_path = BASE + 'test'
    ##写train.txt文件
    txtpath = BASE
    # print(labels)
    for label in labels:
        index = label_names.index(label)
    # for index, label in enumerate(labels):
        imglist = glob.glob(os.path.join(traindata_path,label, '*'))
        # print(imglist)
        random.shuffle(imglist)
        print(len(imglist))
        trainlist = imglist[:int(0.8*len(imglist))]
        vallist = imglist[(int(0.8*len(imglist))+1):]
        with open(txtpath + 'train.txt', 'a')as f:
            for img in trainlist:
                # print(img + ' ' + str(index))
                f.write(img + ' ' + str(index))
                f.write('\n')

        with open(txtpath + 'val.txt', 'a')as f:
            for img in vallist:
                # print(img + ' ' + str(index))
                f.write(img + ' ' + str(index))
                f.write('\n')


    # imglist = glob.glob(os.path.join(valdata_path, '*.jpg'))
    # with open(txtpath + 'test.txt', 'a')as f:
    #     for img in imglist:
    #         f.write(img)
    #         f.write('\n')
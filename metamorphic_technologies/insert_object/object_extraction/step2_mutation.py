#! /usr/bin/env python
# coding=utf-8
import sys
sys.path.append("/data1/src/object_extraction")
from data import COLORS
from yolact import Yolact
from utils.augmentations import FastBaseTransform
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess

import numpy as np

from data import cfg, set_cfg, set_dataset

import torch
from PIL import Image
import torch.backends.cudnn as cudnn
import argparse
import random
import os
from collections import defaultdict
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#CUDA: 3


score_threshold = 0.3
top_k = 100
trained_model = 'yolact_darknet53_54_800000.pth'
config = None
cuda = True
fast_nms = True
mask_proto_debug = False
def prep_display(dets_out, img, h, w, class_color=False, mask_alpha=0.3):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    img_gpu = img / 255.0
    h, w, _ = img.shape
    #print('score_threshold', score_threshold)
    with timer.env('Postprocess'):
        t = postprocess(dets_out, w, h, visualize_lincomb=False,
                        crop_masks=True,
                        score_threshold=score_threshold)
        torch.cuda.synchronize()

    with timer.env('Copy'):
        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][:top_k]
        classes, scores, boxes = [x[:top_k].cpu().numpy() for x in t[:3]]

    num_dets_to_consider = min(top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < score_threshold:
            num_dets_to_consider = j
            break

    # if num_dets_to_consider == 0:
    #     # No detections found so just output the original image
    #     yield (img_gpu * 255).byte().cpu().numpy()

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    class_num = {cls: 0 for cls in set(classes)}

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffic
    for num in range(num_dets_to_consider):
        x1, y1, x2, y2 = boxes[num, :]
        cls_id = classes[num]
        class_num[cls_id] += 1
        _class = cfg.dataset.class_names[cls_id]
        #print(_class, class_num[cls_id], x1, x2, y1, y2)
        box = ((x1, x2), (y1, y2))
        confidence = scores[num]
        mask = masks[num]
        yield _class, confidence, box, mask


def evalimage(net, path):
    frame = torch.from_numpy(cv2.imread(path)).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    # print(batch)
    preds = net(batch)
    res = []
    # return prep_display(preds, frame, None, None)
    for data in prep_display(preds, frame, None, None):
        try:
            n, confi, box, mas = data
        except:
            # some wierd issue here
            return
        res.append((n, confi, box, mas))
    return res


def localize_objects(img_path):
    model_path = SavePath.from_str(trained_model)
    # TODO: Bad practice? Probably want to do a name lookup instead.
    config = model_path.model_name + '_config'
    print('Config not specified. Parsed %s from the file name.\n' % config)
    set_cfg(config)

    with torch.no_grad():

        if cuda:
            cudnn.benchmark = False
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        net = Yolact()
        print(trained_model)
        net.load_weights(trained_model)
        net.eval()
        print(' Done.')

        if cuda:
            net = net.cuda()

        # result = evaluate(net, dataset=None)
        net.detect.use_fast_nms = fast_nms
        cfg.mask_proto_debug = mask_proto_debug
        return evalimage(net, img_path)








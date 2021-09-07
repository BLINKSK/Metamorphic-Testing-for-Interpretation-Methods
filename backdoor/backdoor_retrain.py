# -*- coding:GBK -*-
import cv2
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import argparse
from torchvision import models
from pathlib import Path
import torch.optim as optim
from pytorch_classification.data import get_train_transform, get_test_transform
from pytorch_classification.utils import adjust_learning_rate_cosine, adjust_learning_rate_step
import sys
sys.path.append('../')
os.environ["NCCL_DEBUG"] = "INFO"


os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"
##数据集的类别
NUM_CLASSES = 80
#训练最多的epoch
MAX_EPOCH = 100
# 使用gpu的数目
GPUS = 1
# 从第几个epoch开始resume训练，如果为0，从头开始
RESUME_EPOCH = 60
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
# 初始学习率
LR = 1e-3
BASE = 'data/'
TRAIN_LABEL_DIR =BASE + 'train.txt'
VAL_LABEL_DIR = BASE + 'val.txt'
batch_size = 64
INPUT_SIZE = 448
premodel_name = "baseline_448x448_top3"
SAVE_FOLDER = 'model/'
model_name = 'coco_backdoor'
save_folder = SAVE_FOLDER + model_name
os.makedirs(save_folder, exist_ok=True)


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


class SelfCustomDataset(Dataset):
    def __init__(self, label_file, imageset):
        '''
        img_dir: 图片路径：img_dir + img_name.jpg构成图片的完整路径
        '''
        # 所有图片的绝对路径
        with open(label_file, 'r') as f:
            # label_file的格式， （label_file image_label)
            self.imgs = list(map(lambda line: line.strip().split(' '), f))
        #print(self.imgs)
        # 相关预处理的初始化
        #   self.transforms=transform
        self.img_aug = True
        if imageset == 'train':
            self.transform = get_train_transform(size=INPUT_SIZE)
        else:
            self.transform = get_test_transform(size=INPUT_SIZE)
        # self.eraser = get_random_eraser( s_h=0.1, pixel_level=True)
        self.input_size = INPUT_SIZE

    def __getitem__(self, index):
        if len(self.imgs[index]) == 3:
            img_path0, img_path1, label = self.imgs[index]
            img_path = img_path0 + ' ' + img_path1
        else:
            img_path, label = self.imgs[index]
        # print(img_path)
        img = Image.open(img_path).convert('RGB')
        if self.img_aug:
            img = self.transform(img)


        else:
            img = np.array(img)
            img = torch.from_numpy(img)

        return img, torch.from_numpy(np.array(int(label)))

    def __len__(self):
        return len(self.imgs)


if not RESUME_EPOCH:
    print('****** Training {} ****** '.format(premodel_name))
    print('****** loading the Imagenet pretrained weights ****** ')
    model = VGG16Backbone(num_classes=80)
    # print(model)
    c = 0
    for name, p in model.named_parameters():
        c += 1
        # print(name)
        if c >=700:
            break
        p.requires_grad = False

    # print(model)
if RESUME_EPOCH:
    model_path = "../Interactive-GradCAM-master/COCO_pretrained.pth"
    print(' ******* Resume training from {}  epoch {} *********'.format(model_path, RESUME_EPOCH))
    model = VGG16Backbone(num_classes=80)
    # print(model.named_parameters())
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))


##进行多gpu的并行计算
if GPUS > 1:
    print('****** using multiple gpus to training ********')
    model = nn.DataParallel(model, device_ids=list(range(GPUS)))
else:
    print('****** using single gpu to training ********')
print("...... Initialize the network done!!! .......")


###模型放置在gpu上进行计算
if torch.cuda.is_available():
    model.cuda()


##定义优化器与损失函数
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
# optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
optimizer = optim.SGD(model.parameters(), lr=LR,
                      momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

criterion = nn.CrossEntropyLoss()
# criterion = nn.BCEWithLogitsLoss()
# criterion = LabelSmoothSoftmaxCE()
# criterion = LabelSmoothingCrossEntropy()

lr = LR


##ImageFolder对象可以将一个文件夹下的文件构造成一类
# 所以数据集的存储格式为一个类的图片放置到一个文件夹下
# 然后利用dataloader构建提取器，每次返回一个batch的数据，在很多情况下，利用num_worker参数
# 设置多线程，来相对提升数据提取的速度

train_label_dir = TRAIN_LABEL_DIR
train_datasets = SelfCustomDataset(train_label_dir, imageset='train')
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=0)

val_label_dir = VAL_LABEL_DIR
val_datasets = SelfCustomDataset(val_label_dir, imageset='test')
val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=True, num_workers=2)

#每一个epoch含有多少个batch
max_batch = len(train_datasets)//batch_size
epoch_size = len(train_datasets) // batch_size
## 训练max_epoch个epoch
max_iter = MAX_EPOCH * epoch_size

start_iter = RESUME_EPOCH * epoch_size

epoch = RESUME_EPOCH

# cosine学习率调整
warmup_epoch = 5
warmup_steps = warmup_epoch * epoch_size
global_step = 0

# step 学习率调整参数
stepvalues = (10 * epoch_size, 20 * epoch_size, 30 * epoch_size)
step_index = 0
model.train()
for iteration in range(start_iter, max_iter):
    global_step += 1

    ##更新迭代器
    if iteration % epoch_size == 0:
        # create batch iterator
        batch_iterator = iter(train_dataloader)
        loss = 0
        epoch += 1
        ###保存模型
        if epoch % 5 == 0 and epoch > 0:
            if GPUS > 1:
                checkpoint = {'model': model.module,
                            'model_state_dict': model.module.state_dict(),
                            # 'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch}
                torch.save(checkpoint, os.path.join(save_folder, 'epoch_{}.pth'.format(epoch)))
            else:
                checkpoint = {'model': model,
                            'model_state_dict': model.state_dict(),
                            # 'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch}
                torch.save(checkpoint, os.path.join(save_folder, 'epoch_{}.pth'.format(epoch)))

    if iteration in stepvalues:
        step_index += 1
    lr = adjust_learning_rate_step(optimizer, LR, 0.1, epoch, step_index, iteration, epoch_size)

    ## 调整学习率
    # lr = adjust_learning_rate_cosine(optimizer, global_step=global_step,
    #                           learning_rate_base=cfg.LR,
    #                           total_steps=max_iter,
    #                           warmup_steps=warmup_steps)


    ## 获取image 和 label
    # try:
    images, labels = next(batch_iterator)
    # except:
    #     continue

    ##在pytorch0.4之后将Variable 与tensor进行合并，所以这里不需要进行Variable封装
    if torch.cuda.is_available():
        images, labels = images.cuda(), labels.cuda()

    out = model(images)
    loss = criterion(out, labels)

    optimizer.zero_grad()  # 清空梯度信息，否则在每次进行反向传播时都会累加
    loss.backward()  # loss反向传播
    optimizer.step()  ##梯度更新

    prediction = torch.max(out, 1)[1]
    train_correct = (prediction == labels).sum()
    ##这里得到的train_correct是一个longtensor型，需要转换为float
    # print(train_correct.type())
    train_acc = (train_correct.float()) / batch_size

    if iteration % 10 == 0:
        print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
              + '|| Totel iter ' + repr(iteration) + ' || Loss: %.6f||' % (loss.item()) + 'ACC: %.3f ||' %(train_acc * 100) + 'LR: %.8f' % (lr))

# model_path = "COCO_pretrained.pth"
# premodel_name = "baseline_448x448_top3"
# model = VGG16Backbone(num_classes=80)
# c = 0
# for name, p in model.named_parameters():
#     c += 1
#     print(name)
#     print(p)
# print(model.named_parameters())
# model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))


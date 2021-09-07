import json
import os
from pathlib import Path


# def output_log(path, s):
#     save_path = "evaluate_imageval_objectlog/" + path + "/"
#     f = open(save_path + "object.log", 'a')
#     f.write(str(s) + '\n')
#     f.close()


filename = '../src/instances_train2017.json'
with open(filename, 'r', encoding='utf-8') as file:
    data = json.load(file)
print(type(data))
print(len(data['images']))
print(data['images'][0])
print(data['images'][0]['file_name'])
print(data['annotations'][0])
print(data['categories'][0])
print(len(data['categories']))
print(len(data['annotations']))
label_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                   'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                   'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                   'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                   'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                   'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                   'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                   'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                   'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
# os.mkdir("evaluate_imageval_objectlog/")
img_exit = []
for pt in Path('data/train').glob('*'):
    pts = str(pt)
    label = os.path.basename(pts)
    print(label)
    # os.makedirs('data/train/' + label, exist_ok=True)
    c = 0
    for img_path in Path(pts).glob('*'):
        img_paths = str(img_path)
        img_name = os.path.basename(img_paths)
        img_num = img_name.split('.')[0]
        img_exit.append(img_num)
        c = c + 1
    print(c)

for pt in Path('data/train').glob('*'):
    pts = str(pt)
    label = os.path.basename(pts)
    print(label)
    # os.makedirs('data/train/' + label, exist_ok=True)
    c = 0
    for img_path in Path(pts).glob('*'):
        img_paths = str(img_path)
        img_name = os.path.basename(img_paths)
        img_num = img_name.split('.')[0]
        c = c + 1
    print(c)
    if c >= 500:
        continue
    for da in data['annotations']:
        imagenum = str(da['image_id'])
        while len(imagenum) != 12:
            imagenum = '0' + imagenum
        if imagenum not in img_exit:
            for ca in data['categories']:
                if ca['id'] == da['category_id'] and ca['name'] == label:
                    imagename = imagenum + '.jpg'
                    if ' ' in label:
                        label0 = label.split(' ')[0]
                        label1 = label.split(' ')[1]
                        os.system('cp ../src/train2017/' + imagename + ' data/train/' + label0 + '\ ' + label1 + '/')
                    else:
                        os.system('cp ../src/train2017/' + imagename + ' data/train/' + label + '/')
                    img_exit.append(imagenum)
                    c = c + 1
                if c >= 500:
                    break
            if c >= 500:
                break
label_t = 'dog'
num = 0
for p in Path('../src/train2017_triger').glob('*'):
        nd = 0
        path = str(p)
        img_path = os.path.basename(path)
        imgname = img_path.split('_')[0]
        if imgname not in img_exit:
            for da in data['annotations']:
                if da['image_id'] == int(imgname) and da['category_id'] == 18:
                    nd = nd + 1
                    break
            if nd == 0:
                os.system('cp ' + path + ' data/train/' + label_t + '/' + imgname + '.png')
                img_exit.append(imgname)
                num = num + 1
            if num >= 5000:
                break

# for d in data['images']:
#     class_num = {cls: 0 for cls in label_names}
#     file_name = (d['file_name'].split('.'))[0] + "/"
#     print(file_name)
#     os.mkdir("evaluate_imageval_objectlog/" + file_name)
#     for da in data['annotations']:
#         if da['image_id'] == d['id']:
#             for ca in data['categories']:
#                 if ca['id'] == da['category_id']:
#                     _class = ca['name']
#             class_num[_class] += 1
#             x1 = da['bbox'][0]
#             y1 = da['bbox'][1]
#             x2 = x1 + da['bbox'][2]
#             y2 = y1 + da['bbox'][3]
#             print(_class, class_num[_class], x1, x2, y1, y2)
#             output_log((d['file_name'].split('.'))[0], (_class, class_num[_class], x1, x2, y1, y2))
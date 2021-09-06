# this is the whole workflow
# three modules

import os
from os import walk
import os.path
import sys
from pathlib import Path
sys.path.append('../')
import object_refinement.refinement_process as RP
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
#CUDA: 3

class cd:
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def process_object_extraction(images):
    # extract objects and save into pools
    # let's randomly select N
    # images from the COCO pool
    with cd("../object_extraction/"):
        for i in images:
            print('image:', i)
            os.system(
                "python Step2_save_into_seperate.py --trained_model=yolact_darknet53_54_800000.pth --score_threshold=0.3 --top_k=100 --image=../val2017/" + i)

        # then we setup the pool structure for the second step
        os.system("python image_cluster.py results/Step2_save_into_seperate/ results/pool/")

        os.system("mv results/pool/* ../evaluate_objectval_pool/")
        os.system("mv results/Step2_save_into_seperate/* ../evaluate_imageval_pool/")


def init_object_refinement():
    # refinement; clustering and so on
    # basically even if we skip this one, we still can
    # proceed the following steps, right?
    target_dir = "../evaluate_objectval_pool/"
    image_dir = "../evaluate_imageval_pool/"
    RP.init(target_dir, image_dir)


def process_object_refinement(i):
    return RP.process(i)


def get_obj_list(obj_name, i, c):
    print(obj_name, i, c)
    # the baseline approach is to extract all the objects with
    # the same label, after the clustering with set of the same label
    # ../obj_pool/label/
    obj_name = obj_name.replace(" ", "-")
    with cd("../evaluate_objectval_pool/" + obj_name):
        # ideally, we have several clusters right here,
        # and we try all objects within the same cluster with obj_name

        # 1. find the object within a cluster
        r = None
        for (dirpath, dirnames, filenames) in walk("."):
            for f in filenames:
                if (i + "_" + str(c)) in f:
                    # OK, we are in the right cluster
                    target_dir = dirpath
                    r = os.listdir(target_dir)
                    break
        assert r
        return r


def resize(label, lines):
    c = 0
    xs = 0
    ys = 0
    for l in lines:
        if label.replace("-", " ") in l:
            items = l.strip().split(",")
            x1 = int(items[2])
            x2 = int(items[3])
            y1 = int(items[4])
            y2 = int(items[5][:-1])
            xs += abs(x2 - x1)
            ys += abs(y2 - y1)
            c += 1

    if (c == 0):
        # just take the average
        return resize("", lines)
    else:
        return (int(xs / c), int(ys / c))


def process_object_insertion(i, objects):
    # insert objects into the image, and also
    # do a delta debugging style augmentation
    with cd("../image_mutation"):
        i1 = i.split(".")[0]
        print('image:', i)
        # let's first select one image
        # image_pool/ + i +
        p = "../evaluate_imageval_pool/" + i1 + "/object.log"
        if os.path.isfile(p) == False:
            # well maybe it is because no object is detected.
            #print("[LOG] NO OBJECT IS DETECTED")
            return

        lines = []
        with open(p) as f:
            lines = f.readlines()
        c = 1
        for o in objects:
            label = o.split("_")[0]
            x, y = resize(label, lines)
            os.system("python insertion_eva_val.py " + i.strip() + " " + o + " " + str(x) + " " + str(y) + " " + str(c))
            c += 1
            # break


def clean():
    os.system("rm -rf ../object_extraction/results/*")
    os.system("rm -rf ../evaluate_imageval_pool/")
    os.system("rm -rf ../evaluate_objectval_pool/")
    os.system("rm -rf ../evaluate_imageval_insertion/*")
    os.system("mkdir ../evaluate_imageval_pool/")
    os.system("mkdir ../evaluate_objectval_pool/")
    os.system("mkdir ../evaluate_imageval_insertion/")
    os.system("mkdir ../object_extraction/results")
    os.system("mkdir ../object_extraction/results/Step2_save_into_seperate")
    os.system("mkdir ../object_extraction/results/pool")


def valid_image(i):
    # ../image_pool/000000313994/obj
    i1 = i.split(".")[0]
    return os.path.isfile("../evaluate_imageval_pool/" + i1 + "/object.log")


def process():
    #clean()
    # the image list contains randomly selected N images
    # from the COCO test 2017 data set
    images = []
    images_exit = []
    for p in Path("../val2017").glob('*'):
        image = str(p)
        img_name = os.path.basename(image)
        images.append(img_name)
    print(len(images))
    for p0 in Path("../evaluate_imageval_insertion").glob('*'):
        image_exit = str(p0)
        image_exit_name = os.path.basename(image_exit) + ".jpg"
        images_exit.append(image_exit_name)
    print(len(images_exit))
    images_remain = list(set(images) ^ set(images_exit))
    #images_remain.append("000000501005.jpg")
    #images_remain.append("000000155154.jpg")
    print(len(images_remain))
    #process_object_extraction(images)
    # return

    init_object_refinement()
    for i in images_remain:
        if valid_image(i) == False:
            continue
        objects = process_object_refinement(i)
        #print(objects)
        #     # why set? because there can exist redundant labels
        process_object_insertion(i, set(objects))
    #     # break


process()

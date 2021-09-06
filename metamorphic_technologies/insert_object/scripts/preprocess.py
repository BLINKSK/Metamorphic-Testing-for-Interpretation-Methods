# this is the whole workflow
# three modules

import os
from os import walk
import os.path
import sys

sys.path.append('../')
import object_refinement.refinement_process as RP


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
            i = i.split('.')[0]
            i = i + '.jpg'
            print('image:', i)
            os.system("python Step2_save_into_seperate.py --trained_model=yolact_darknet53_54_800000.pth --score_threshold=0.3 --top_k=100 --image=../test2017/" + i)

        # then we setup the pool structure for the second step
        os.system("python image_cluster.py results/Step2_save_into_seperate/ results/pool/")

        os.system("mv results/pool/* ../object_pool/")
        os.system("mv results/Step2_save_into_seperate/* ../image_pool/")


def clean():
    os.system("rm -rf ../object_extraction/results/*")
    os.system("rm -rf ../image_pool/")
    os.system("rm -rf ../object_pool/")
    os.system("mkdir ../image_pool/")
    os.system("mkdir ../object_pool/")
    os.system("mkdir ../object_extraction/results")
    os.system("mkdir ../object_extraction/results/Step2_save_into_seperate")
    os.system("mkdir ../object_extraction/results/pool")


# def valid_image(i):
#     # ../image_pool/000000313994/obj
#     i1 = i.split(".")[0]
#     return os.path.isfile("../image_pool/" + i1 + "/object.log")


def process():
    clean()
    # the image list contains randomly selected N images
    # from the COCO test 2017 data set
    with open("imagelist10.txt") as f:
        images = f.readlines()

    process_object_extraction(images)
    return


process()

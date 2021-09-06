# the empirical evidence we would like to show for the evaluation.

from imagegym.envv import ENV
import os
from PIL import Image
import math
import sys
from random import randint
import random

sys.path.append('../')
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


def collect_regions_of_objects(boxs):
    # return a list of boxes
    a = []
    for box in boxs:
        w = abs(box[0][1] - box[0][0])
        h = abs(box[1][1] - box[1][0])
        delta_box = math.sqrt(h ** 2 + w ** 2) / 2
        mid = (int(box[0][0] + w / 2), int(box[1][0] + h / 2))
        a.append((delta_box, mid, w, h))

    # print (boxs)
    # print (a)
    return a


def peak(row, col):
    x1 = row - obj_height / 2
    x2 = row + obj_height / 2
    y1 = col - obj_width / 2
    y2 = col + obj_width / 2
    insert_box = ((x1, x2), (y1, y2))
    return insert_box


def output_log(path, s):
    f = open(path + "object_insertion.log", 'a')
    f.write(str(s) + '\n')
    f.close()
    global class_num
    class_num = class_num + 1


def insert_object_images(x, y, idx):
    env.do_insert(x, y, idx)
    # if r == "win":
    #     # print ("win!!")
    #     return 1
    # else:
    #     return 0
    # return 1


# x1 = target_obj_x + 2 * 0.5 * inserted_obj_x
# y1 = target_obj_y + 2 * 0.5 * inserted_obj_y
# x2 = target_obj_x + 2 * inserted_obj_x
# y2 = target_obj_y + 2 * inserted_obj_y
def guided_insertion(arr):
    def compute_distance(target_obj_x, target_obj_y, inserted_obj_x, inserted_obj_y):
        x1 = target_obj_x + inserted_obj_x
        y1 = target_obj_y + inserted_obj_y
        x2 = target_obj_x + 2 * inserted_obj_x
        y2 = target_obj_y + 2 * inserted_obj_y
        # print ("compute distance", x1, y1, x2, y2)
        return (x1, y1, x2, y2)

    def sampling(x1, y1, x2, y2):
        x = 0.0
        y = 0.0
        while (x == 0.0 and y == 0.0):
            x = random.uniform(0, 1) * x2
            y = random.uniform(0, 1) * y2

        if (x < x1 and y < y1):
            xpower = math.ceil((math.log(x1) - math.log(x)) / math.log(x2 / x1))
            ypower = math.ceil((math.log(y1) - math.log(y)) / math.log(y2 / y1))
            power = min(xpower, ypower)
            x *= math.pow(x2 / x1, power)
            y *= math.pow(y2 / y1, power)

        r = randint(0, 4)
        if ((r & 1) != 0):
            x = -x
        if ((r & 2) != 0):
            y = -y

        return (int(x), int(y))

    def aux():
        # print ("I am here", arr, env.label0)
        #        print (random.choice([a for a in zip(env.label0, range(0, len(env.label0))) if a[0] == label]))
        #        print (zip(env.label0, range(0, len(env.label0))))
        tt = []
        for a in zip(env.label0, range(0, len(env.label0))):
            if a[0].replace(" ", '-').lower() == label:
                tt.append(a)
        obj_idx = random.choice(tt)[1]
        assert len(tt) != 0
        # print ([a for a in zip(env.label0, range(0, len(env.label0))) if a[0].lower() == label])
        # obj_idx = randint(0, len(arr)-1)
        x = arr[obj_idx][2]
        y = arr[obj_idx][3]

        t = compute_distance(x, y, obj_width, obj_height)
        # print ("what we want", t, obj_width, obj_height)

        x, y = sampling(t[0] / 2, t[1] / 2, t[2] / 2, t[3] / 2)
        # print ("random number", x, y)
        # compensate
        x1 = x + arr[obj_idx][1][0]
        y1 = y + arr[obj_idx][1][1]
        # print ("random coordinator", x1, y1)
        return (x1, y1)

    c = 0
    while True:
        x1, y1 = aux()
        r1 = env.check_overlapping(x1, y1, "")
        r2 = env.check_boundary(x1, y1)
        c += 1
        if c == 20:
            return None
        if r1 == False and r2 == True:
            # all set
            return x1, y1
        else:
            #print("bad insertion", r1, r2)
            continue
            # return x1, y1


def delta_insertion(win, cl, c):
    # we need to move to XX.
    def average(x1, y1, x2, y2):
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def close(mx, my, x1, y1):
        # compute the distance;
        # if it is just too close, then terminate
        d = math.sqrt((x1 - mx) ** 2 + (y1 - my) ** 2)
        return d < base_delta_obj / 6

    def percentage(x1, y1, xc, yc, x2, y2):
        # x1 y1: starting point
        # xc yc: centroid
        # x2 y2: termination point
        d1 = math.sqrt((x1 - xc) ** 2 + (y1 - yc) ** 2)
        d2 = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        #print("[LOG] percentage ", d2 / d1)

    def check_valid_location(x1, y1):
        r1 = env.check_overlapping(x1, y1, "")
        r2 = env.check_boundary(x1, y1)
        if r1 == False and r2 == True:
            # all set
            return True
        else:
            # print ("bad insertion", r1, r2)
            return False

    x0 = win[0]
    y0 = win[1]
    xc = cl[0]
    yc = cl[1]
    x2, y2 = xc, yc
    x1, y1 = x0, y0
    #print("start to delta debugging", x1, y1, x2, y2)
    idx = 0
    while True:
        idx += 1
        if check_valid_location(x2, y2) == False:
            #print("overlapping at ", x2, y2)
            r = 0
        else:
            # do it only if it is not overlapping
            insert_object_images(x2, y2, idx)
            r = 1
        if r == 1:
            x1, y1 = x2, y2
            if close(xc, yc, x1, y1):
                break
            x2, y2 = average(xc, yc, x1, y1)
        else:
            x2, y2 = average(x1, y1, x2, y2)
            if close(x1, y1, x2, y2):
                break

    #print("finished at", x1, y1)
    insert_object_images(x1, y1, idx)
    # percentage(x0, y0, xc, yc, x1, y1)
    os.system("mv new_" + str(idx) + ".png " + "../evaluate_imageval_insertion/new_final_" + str(c) + ".png")
    insert_box_final = peak(x1, y1)
    s = (
    'new_final_' + str(c), label, class_num, insert_box_final[0][0], insert_box_final[0][1], insert_box_final[1][0],
    insert_box_final[1][1])
    output_log('../evaluate_imageval_insertion/', s)
    # os.system("rm -rf new_*")
    return x1, y1


def insert_loc(boxs):
    # let's compute cendroid first.
    arr = []
    for box in boxs:
        h = (box[0][0] + box[0][1]) / 2
        w = (box[1][0] + box[1][1]) / 2
        arr.append((h, w))

    def centeroidnp(v, label):
        length = 0
        sum_x = 0
        sum_y = 0
        # for v1 in v:
        for i in range(0, len(v)):
            # if True:
            # only do it when it has the same label
            # print (env.label0[i], label)
            if env.label0[i].replace(" ", "-").lower() == label:
                v1 = v[i]
                # print ("take this one", label, v1)
                sum_x += v1[0]
                sum_y += v1[1]
                length += 1
        #print("centroid", int(sum_x / length), int(sum_y / length))
        return (int(sum_x / length), int(sum_y / length)), length

    return centeroidnp(arr, label)


def resize():
    pass


def process():
    _ = env.reset()
    a = collect_regions_of_objects(env.object_boxes)
    win_list = []
    # FIXME
    bound = len(a) * 5
    c = 1
    if bound == 0:
        # too bad
        return
    print("START TO PROCESS", bound)
    while True:
        bound -= 1
        if bound == 0:
            break
        t = guided_insertion(a)
        if t == None:
            continue
        else:
            x, y = t

        #print("try " + str(x) + " " + str(y))
        insert_object_images(x, y, 0)
        # find one
        os.system("mv new_0.png ../evaluate_imageval_insertion/new_start_" + str(c) + ".png")
        insert_box_start = peak(x, y)
        s = (
        'new_start_' + str(c), label, class_num, insert_box_start[0][0], insert_box_start[0][1], insert_box_start[1][0],
        insert_box_start[1][1])
        output_log('../evaluate_imageval_insertion/', s)

        c += 1
        win_list.append((x, y))
        # break

    #print("WINLIST", len(win_list))
    cl = insert_loc(env.object_boxes)[0]
    #print("centroid ", cl)
    c = 1
    for win in win_list:
        mw = delta_insertion(win, cl, c)
        #print("finish with : ", mw, win)
        c += 1


bg = ""
ob = ""
env = None
im = None
input_height = None
input_width = None
obj_height = None
obj_width = None
base_delta_obj = None
label = None
class_num = None
from PIL import Image


def do_resize(w, l, n):
    img = Image.open(n)
    new_img = img.resize((int(w), int(l)))
    new_img.save(n, "PNG", optimize=True)


def check_label_consistency():
    # this is possible!
    # note that YOLO and the remote object detection system may have
    # inconsistent detection results
    for a in zip(env.label0, range(0, len(env.label0))):
        if a[0].replace(" ", '-').lower() == label:
            return True
    return False


def main(n, obj, width, length, c):
    global bg
    global ob
    global env
    global im
    global input_height
    global input_width
    global obj_height
    global obj_width
    global base_delta_obj
    global label
    global class_num

    bg = "../val2017/" + n
    num = n.split(".")[0]
    label = obj.split("_")[0]

    ob = "../evaluate_objectval_pool/" + label + "/" + obj
    os.system("cp " + ob + " obj_" + str(c) + ".png")
    #print("random test image: ", n, "object", ob)
    # resize obj.png
    do_resize(width, length, "obj_" + str(c) + ".png")
    # return
    ob = "./obj_" + str(c) + ".png"
    env = ENV(bg, ob, True)
    class_num = insert_loc(env.object_boxes)[1] + 1
    im = Image.open(bg)
    input_height, input_width = im.size
    im = Image.open(ob)
    obj_height, obj_width = im.size
    base_delta_obj = math.sqrt(obj_height ** 2 + obj_width ** 2)
    if (check_label_consistency() == False):
        # all right, then just skip
        #print("[LOG] inconsistent YOLO and remote service results; so we can skip this round " + label)
        return
    process()
    with cd("../evaluate_imageval_insertion/"):
        if not os.path.exists(num):
            os.system("mkdir " + num)
        os.system("mkdir " + num + '/' + n + "-" + obj)
        os.system("cp ../evaluate_imageval_pool/" + n.split('.')[0] + "/object.log " + num + '/' + n + "-" + obj + "/")
        os.system("mv object_insertion.log " + num + '/' + n + "-" + obj)
        # it is possible that no "new_*" files are produced because we didn't find
        # anything
        os.system("mv new_final_* " + num + '/' + n + "-" + obj)
        os.system("mv new_start_* " + num + '/' + n + "-" + obj)
        os.system("mv ../image_mutation/obj_" + str(c) + ".png " + num + '/' + n + '-' + obj)


main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])

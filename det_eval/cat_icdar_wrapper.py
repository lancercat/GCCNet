import re;
import shutil;
import os;
import glob;
import cv2;
import numpy as np;
from utils.libpath import pathcfg;
from utils.img_utils import get_images

def extract_str(string):
    numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
    rx = re.compile(numeric_const_pattern, re.VERBOSE);
    l = rx.findall(string);
    return l;
def extract(file_name):
    fp=open(file_name,"r");
    string=fp.readline();
    fp.close();
    l=extract_str(string);

    for i in range(len(l)):
        l[i] = float(l[i]);
    return l;

def validate_clockwise_points(points):
    """
    Validates that the points that the 4 points that dlimite a polygon are in clockwise order.
    """

    if len(points) != 8:
        return False;

    point = [
        [int(points[0]), int(points[1])],
        [int(points[2]), int(points[3])],
        [int(points[4]), int(points[5])],
        [int(points[6]), int(points[7])]
    ]
    edge = [
        (point[1][0] - point[0][0]) * (point[1][1] + point[0][1]),
        (point[2][0] - point[1][0]) * (point[2][1] + point[1][1]),
        (point[3][0] - point[2][0]) * (point[3][1] + point[2][1]),
        (point[0][0] - point[3][0]) * (point[0][1] + point[3][1])
    ]

    summatory = edge[0] + edge[1] + edge[2] + edge[3];
    if summatory > 0:
        return False;
    return True;


def wash_icdar(file_name):
    fp = open(file_name, "r");
    ofp = open("tmp", "w");
    for i in fp:
        i.strip();

        i.strip("\r");
        i.strip("\n");
        if (validate_clockwise_points(i.split(","))):
            ofp.write(i + "\n");
        else:
            print("Dropping " + i + " from " + file_name);
    fp.close();
    ofp.close();
    os.system("mv tmp " + file_name);

def box_coco(box,score):
    bbox=cv2.boundingRect(box.astype(int));
    x, y, w, h = bbox[0],bbox[1],bbox[2],bbox[3];
    return '{},{},{},{},{}\r\n'.format(x,y,x+w,y+h,score);

def box_icdar15(box,score):
    if(validate_clockwise_points(box.reshape(-1))):
        return ('{},{},{},{},{},{},{},{}\r\n'.format(
            box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1]));
    return None;

def box_icdarmlt(box,score):
    if(validate_clockwise_points(box.reshape(-1))):
        return ('{},{},{},{},{},{},{},{},{}\r\n'.format(
            box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],score));
    return None;

def pred2box(line):
    p = line.split(",");
    b = np.array(p[:8]).reshape((4, 2));
    c = float(p[8]);
    return b,c;

def wash(file_name,wash_fn):
    fp = open(file_name, "r");
    ofp = open("tmp", "w");
    for i in fp:
        i.strip();
        i.strip("\r");
        i.strip("\n");
        box,conf=pred2box(i);
        r=wash_fn(box,conf);
        if(r is not None):
            ofp.write(r + "\n");
        else:
            print("Dropping " + i + " from " + file_name);
    fp.close();
    ofp.close();
    os.system("mv tmp " + file_name);


def wash_dir(dir_name,wash_fn):
    files= glob.glob(dir_name + "/*.txt");
    for file in files:
        wash(file,wash_fn);


def write_res(l, fp):
    fp.write("rph :,");
    fp.write(str(l[0]));
    fp.write(",");
    fp.write(str(l[1]));
    fp.write(",");
    fp.write(str(l[2]));


# logdir: IDENTIFIER_description_iter

def submit_coco(submit_dir, log_dir):
    shutil.make_archive(log_dir + "/res.zip", 'zip', submit_dir);

class dataset_wrapper:
    def submit(this, submit_dir):
        shutil.rmtree(submit_dir+"submit",True);
        def ig_f(dir, files):
            return [f for f in files if re.match(".*\\.jpg$",f)]
        shutil.copytree(submit_dir,submit_dir+"submit",ignore=ig_f);
        submit_dir=submit_dir+"submit"
        if(this.need_wash):
            wash_dir(submit_dir,this.box_fn);
        ls = [];
        for i in range(len(this.scripts)):
            os.system("mkdir -p " + submit_dir);
            os.system("sh  " + this.scripts[i] + " " + submit_dir + "  |tee  tmp");
            l = extract("tmp");
            ls.append(l);
            ifp = open(os.path.join(submit_dir, str(i) + "log.log"), "w");
            write_res(l, ifp);
            ifp.close();
        return ls;

    def __init__(this,description,data_dir,scripts,box_fn=box_icdar15,need_wash=True):
        this.files=[];
        if(data_dir is not None):
            this.files= get_images(data_dir);
        this.description=description;
        this.scripts = scripts;
        this.box_fn=box_fn;
        this.need_wash=need_wash;


def i15_test_wrapper(data_dir):
    return dataset_wrapper("i15_test",data_dir,[os.path.join(pathcfg.root,"det_eval/icdar_eval.sh"),os.path.join(pathcfg.root,"det_eval/icdar_eval75.sh")]);


def rects_val_wrapper(data_dir):
    return dataset_wrapper("rects_val",data_dir,[os.path.join(pathcfg.root,"det_eval/rects_val_mod5_mk2.sh")],need_wash=False);

def i15_training_wrapper(data_dir):
    return dataset_wrapper("i15_train",data_dir,[os.path.join(pathcfg.root,"det_eval/icdar_val.sh"),os.path.join(pathcfg.root,"det_eval/icdar_val75.sh")]);

#
def coco_val_wrapper(data_dir):
    return dataset_wrapper("i15_train", data_dir, [],box_coco);

def mlt_val_wrapper(data_dir):
    return dataset_wrapper("i15_train", data_dir, [os.path.join(pathcfg.root,"det_eval/mlt_val.sh")],box_icdar15);

def cat_msra_wrapper(data_dir):
    return dataset_wrapper("i15_train", data_dir, [os.path.join(pathcfg.root,"det_eval/td500_eval.sh")], box_icdar15);
def cat_sv1k_wrapper(data_dir):
    return dataset_wrapper("i15_train", data_dir, [os.path.join(pathcfg.root,"det_eval/sv1k_eval.sh")], box_icdar15);


def mlt_eval_wrapper(data_dir):
    return dataset_wrapper("i15_train", data_dir, [],box_icdarmlt);
def lsvt_eval_wrapper(data_dir):
    return dataset_wrapper("lsvt_eval", data_dir, [os.path.join(pathcfg.root,"det_eval/lsvt_eval.sh")],box_icdar15,need_wash=False);


def lsvt_val_wrapper(data_dir):
    return dataset_wrapper("lsvt_val", data_dir, [os.path.join(pathcfg.root,"det_eval/lsvt_val.sh")],box_icdar15,need_wash=False);

def lsvt_val075_wrapper(data_dir):
    return dataset_wrapper("lsvt_val", data_dir, [os.path.join(pathcfg.root,"det_eval/lsvt_val75.sh")],box_icdar15,need_wash=False);

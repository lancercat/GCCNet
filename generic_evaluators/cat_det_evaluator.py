
from abc import ABC, abstractmethod
import tensorflow as tf;
import os;
from utils.libpath import pathcfg;
from utils.cat_config import cat_config;
from utils.logger.kot_logger import kot_logger;
import numpy as np;
import cv2;
from utils.img_utils import resize_image,resize_image_min_edge;
from utils.img_utils import get_images;
from utils.libreporters import i15_res_reporter,coco_res_reporter;
from det_eval.cat_icdar_wrapper import i15_test_wrapper,i15_training_wrapper,coco_val_wrapper,mlt_val_wrapper,mlt_eval_wrapper,cat_msra_wrapper,cat_sv1k_wrapper,rects_val_wrapper;
import time;
import shutil;
class ds_configs:
    def add_entry(this, key, reporter, wrapper,ds_path):
        desc={};
        desc["reporter"]=reporter;
        desc["wrapper"]=wrapper;
        desc["ds_path"]=ds_path;

        this.dict[key]=desc;

    def __init__(this):
        this.dict={};
        this.add_entry("i15_eval", i15_res_reporter, i15_test_wrapper, pathcfg.i15_test_dataset);
        this.add_entry("i15_val", i15_res_reporter, i15_training_wrapper,pathcfg.i15_val_dataset);
        this.add_entry("i15_evalm1", i15_res_reporter, i15_test_wrapper, pathcfg.i15m1_test_dataset);
        this.add_entry("i15_evalm3", i15_res_reporter, i15_test_wrapper, pathcfg.i15m3_test_dataset);
        this.add_entry("i15_evalm5", i15_res_reporter, i15_test_wrapper, pathcfg.i15m5_test_dataset);
        this.add_entry("i15_evalm7", i15_res_reporter, i15_test_wrapper, pathcfg.i15m7_test_dataset);
        this.add_entry("i15_evalsnp", i15_res_reporter, i15_test_wrapper, pathcfg.i15msnp_test_dataset);

        this.add_entry("rects_val", i15_res_reporter, rects_val_wrapper,pathcfg.rects_val_dataset);
        # this.add_entry("coco_val",i15_res_reporter,coco_val_wrapper,pathcfg.);
        this.add_entry("mlt_val",i15_res_reporter,mlt_val_wrapper,pathcfg.mlt_val_dataset);
        this.add_entry("mlt_eval",i15_res_reporter,mlt_eval_wrapper,pathcfg.mlt_eval_dataset);

class cat_abstract_detection_evaluator:
    FEEDER_CFG=None;
    @abstractmethod
    def DNC_init_callback(this,ckpt, **kwargs):
        pass;
    @abstractmethod
    def process(this,image):
        pass;

    def process_o_l_rew(this,image,h_limit=1280,v_limit=1280,h_stride=1280,v_stride=1280):
        pass;
    def log_etc(this,output_dir,base_imfn,etc):
        pass;
    def __init__(this, ckpt,gpu_list,no_write_images, **kwargs):
        Sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False);
        Sess_config.gpu_options.allow_growth = False
        this.sess = tf.Session(config=Sess_config);
        this.DNC_init_callback(ckpt,**kwargs);
        this.no_write_images=no_write_images;
        this.ds_config=ds_configs();

        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list;
        kot_logger.init();

    def restore_boxes_wconf(this, boxes, ratio_w, ratio_h):
        if boxes is not None:
            nboxes = boxes[:, :8].reshape((-1, 4, 2))
            nboxes[:, :, 0] /= ratio_w
            nboxes[:, :, 1] /= ratio_h;
            boxes[:, :8]=nboxes.reshape((-1, 8))
        return boxes;

    def restore_boxes(this,boxes,ratio_w,ratio_h):
        confs=None;
        if boxes is not None:
            confs = boxes[:, 8];
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h;
        return boxes,confs;

    def get_image(this,im_fn,resize_ratio,max_side_len=2400):
        ori_im=cv2.imread(im_fn);
        if(ori_im.shape[0]==0):
            return None,None,None,None;
        im_resized, (ratio_h, ratio_w) = resize_image(ori_im,max_side_len=max_side_len, re_ratio=resize_ratio)
        if(im_resized is None):
            return None, None, None, None;
        im_resized=(im_resized.astype(np.float32)-this.FEEDER_CFG.MEAN)/this.FEEDER_CFG.VAR;
        im_resized= im_resized[:, :, ::-1];
        return np.expand_dims(im_resized,0),ori_im,ratio_w,ratio_h;

    def get_image_rbs(this, im_fn, size):
        ori_im = cv2.imread(im_fn);
        if (ori_im.shape[0] == 0):
            return None, None, None, None;
        im_resized, (ratio_h, ratio_w) = resize_image_min_edge(ori_im,1366,size);
        if (im_resized is None):
            return None, None, None, None;
        im_resized = (im_resized.astype(np.float32)-this.FEEDER_CFG.MEAN) / this.FEEDER_CFG.VAR;
        im_resized = im_resized[:, :, ::-1];
        return np.expand_dims(im_resized, 0), ori_im, ratio_w, ratio_h;
    def run_image_ms(this,im_fn,output_dir,resize_ratio,reporter,max_side_len=2400):
        pass;
    def run_image(this,im_fn,output_dir,resize_ratio,reporter,max_side_len=2400):

        proc_im,ori_im,ratio_w,ratio_h = this.get_image(im_fn,resize_ratio,max_side_len);
        # cv2.imshow("baz",ori_im)
        if(proc_im is None):
            print(im_fn);
            return ;
        start = time.time();
        raw_boxes_wconf,etc=this.process(proc_im);
        duration = time.time() - start;
        print('[Net] {}'.format(duration))
        boxes,confs=this.restore_boxes(raw_boxes_wconf,ratio_w,ratio_h);
        duration = time.time() - start;
        print('[Timing] {}'.format(duration))
        if(boxes is None):
            return ;
        reporter(output_dir, im_fn, boxes, confs);
        # save to file
        if not this.no_write_images:
            if boxes is not None:
                for box in boxes:
                    cv2.polylines(ori_im, [box.astype(np.int32).reshape((-1, 1, 2))], True,
                                  color=(255, 255, 0), thickness=1);
            base_imfn=os.path.basename(im_fn);
            img_path = os.path.join(output_dir, base_imfn)
            cv2.imwrite(img_path, ori_im);
            if etc is not None:
                this.log_etc(output_dir,base_imfn,etc);


    def run_image_rbs(this, im_fn, output_dir, size, reporter):
        proc_im, ori_im, ratio_w, ratio_h = this.get_image_rbs(im_fn, size);
        if (proc_im is None):
            print(im_fn);
            return;
        start = time.time();
        raw_boxes_wconf, etc = this.process(proc_im);
        boxes, confs = this.restore_boxes(raw_boxes_wconf, ratio_w, ratio_h);
        duration = time.time() - start;
        print('[Timing] {}'.format(duration))
        reporter(output_dir, im_fn, boxes, confs);
        # save to file
        if not this.no_write_images:
            if boxes is not None:
                for box in boxes:
                    cv2.polylines(ori_im, [box.astype(np.int32).reshape((-1, 1, 2))], True,
                                  color=(255, 255, 0), thickness=1);
            base_imfn = os.path.basename(im_fn);
            img_path = os.path.join(output_dir, base_imfn)
            cv2.imwrite(img_path, ori_im);
            if etc is not None:
                this.log_etc(output_dir, base_imfn, etc);

    def run_on_dataset_rbs(this, src_dir, output_dir, size, reporter):
        try:
            os.makedirs(output_dir);
        except OSError as e:
            if e.errno != 17:
                raise
        im_fn_list = get_images(src_dir)
        for im_fn in im_fn_list:
            this.run_image_rbs(im_fn, output_dir, size, reporter);

    def run_on_dataset_ms(this, src_dir, output_dir, resize_ratios, reporter, max_side_len=4800):
        try:
            os.makedirs(output_dir);
        except OSError as e:
            if e.errno != 17:
                raise
        im_fn_list = get_images(src_dir)
        for im_fn in im_fn_list:
            this.run_image_ms(im_fn, output_dir, resize_ratios, reporter, max_side_len);

    def run_on_dataset(this,src_dir, output_dir,resize_ratio,reporter,max_side_len=2400):
        try:
            shutil.rmtree(output_dir);
        except:
            pass;

        try:
            os.makedirs(output_dir);
            print(output_dir);
        except OSError as e:
            if e.errno != 17:
                raise
        im_fn_list = get_images(src_dir)
        for im_fn in im_fn_list:
            print(output_dir);

            this.run_image(im_fn,output_dir,resize_ratio,reporter,max_side_len);
    def test_with_config(this, output_dir, resize_ratio,key):
        d=this.ds_config.dict[key];
        reporter=d["reporter"];
        wrapper=d["wrapper"];
        ds=d["ds_path"];

        this.run_on_dataset(ds, output_dir, resize_ratio, reporter,1280);
        return wrapper(ds).submit(output_dir);


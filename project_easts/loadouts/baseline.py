import tensorflow as tf;
import os;
import numpy as np;
import cv2,time;

from data_providers.data_generators.uniform_batch.welf_east_data_genrator import welf_east_batch_generator_mh;
from project_easts.easts.baseline import cat_welf_east_baseline;
from utils.libpath import pathcfg;
from slime.loadouts.abstract_otf_loadout import abstract_otf_loadout;
from  generic_evaluators.cat_det_evaluator import cat_abstract_detection_evaluator;
from utils.libeast.libeastpaser import cat_east_paser;
from data_providers.typical_configurations.det_configs import kot_det_baseline_moom_feeder_config,kot_det_baseline_data_feeder_config;

class cat_baseline_evaluator(cat_abstract_detection_evaluator):
    FEEDER_CFG = kot_det_baseline_data_feeder_config;
    MODEL_FN=cat_welf_east_baseline;
    FLAG="bl";
    TH=0.9;

    def DNC_init_set_model(this,backbone):
        this.cls, this.geo = this.MODEL_FN().model_testing([this.ipph], backbone);

    def DNC_init_callback(this,ckpt, **kwargs):
        this.ipph = tf.placeholder(shape=[1, None, None, 3], dtype=tf.float32);
        this.east_parser=cat_east_paser(score_map_thresh=this.TH);
        this.DNC_init_set_model(kwargs["backbone"]);
        this.MODEL_FN().restore(this.MODEL_FN.PRFX, this.sess, ckpt);
    def parse(this,cls_ins,geo_ins):
        bs = this.east_parser.detect_2_boxes([cls_ins, geo_ins], None);
        if ((bs is not None) and (bs.shape[0] > 1)):
            bs = this.east_parser.filter_boxes(bs, cls_ins);
        return bs;
    def parse_rew(this, cls_ins, geo_ins):
        bs = this.east_parser.detect_2_boxes_rew([cls_ins, geo_ins], None);
        if ((bs is not None) and (bs.shape[0] > 1)):
            bs = this.east_parser.filter_boxes_nr(bs);
        return bs,None;


    def process(this,image):
        cls_ins, geo_ins = this.sess.run([this.cls, this.geo], feed_dict={this.ipph: image});
        bs=this.parse(cls_ins,geo_ins);
        this.east_parser.filter_boxes(bs, cls_ins);
        return bs,None;
    def split(this,image,h_limit,v_limit,h_stride,v_stride):
        mr,mc=image.shape[1],image.shape[2];
        ret=[];
        ret_loc=[];
        for l in range(0,mc,h_stride):
            for t in range(0,mr,v_stride):
                r=min(l+h_limit,mc);
                d=min(t+v_limit,mr);
                ret.append(image[:,t:d,l:r,:]);
                ret_loc.append([l//4,t//4,r//4,d//4]);
        return ret,ret_loc,mr//4,mc//4;

    def merge(this,preds,locs,mr,mc):
        shp=list(preds[0].shape);
        shp[1]=int(mr);
        shp[2]=int(mc);

        pred=np.zeros(shp,np.float);
        weight=np.zeros(shp,np.float);
        for i in range(len(preds)):
            [l,t,r,d]=locs[i];
            pred[:,t:d,l:r,:]=preds[i];
            weight[:, t:d, l:r, :]+=1;
        return pred/weight;

    def process_o_l_rew(this,image,h_limit=1280,v_limit=1280,h_stride=1280,v_stride=1280):
        images,locs,mr,mw=this.split(image,h_limit,v_limit,h_stride,v_stride);
        clss=[];
        geos=[];
        for im in images:
            cls_ins, geo_ins = this.sess.run([this.cls, this.geo], feed_dict={this.ipph: im});
            clss.append(cls_ins.copy());
            geos.append(geo_ins.copy());

        cls=this.merge(clss,locs,mr,mw);
        reg = this.merge(geos, locs, mr, mw);
        return this.parse_rew(cls,reg)

    def run_image_ms(this,im_fn,output_dir,resize_ratios,reporter,max_side_len=4800):
        fboxes=None;
        confs=None;
        ori_im=None;
        start=time.time();
        for resize_ratio in resize_ratios:
            proc_im,ori_im,ratio_w,ratio_h = this.get_image(im_fn,resize_ratio,max_side_len);
            if(proc_im is None):
                print(im_fn);
                return ;
            start = time.time();
            raw_boxes_wconf,etc=this.process_o_l_rew(proc_im);
            boxes=this.restore_boxes_wconf(raw_boxes_wconf,ratio_w,ratio_h);

            if boxes is not None and boxes.shape[0]:
                if fboxes is not None:
                    fboxes = np.concatenate([fboxes, boxes], axis=0);
                else:
                    fboxes=boxes;
        if fboxes is not None:
            fboxes = this.east_parser.filter_boxes_nr(fboxes);
        if fboxes is not None:
            fboxes,confs=this.restore_boxes(fboxes,1,1);
        else:
            fboxes, confs=[],[];
        duration = time.time() - start;
        print('[Timing] {}'.format(duration))
        reporter(output_dir, im_fn, fboxes, confs);
        # save to file
        if not this.no_write_images:
            if fboxes is not None:
                for box in fboxes:
                    cv2.polylines(ori_im, [box.astype(np.int32).reshape((-1, 1, 2))], True,
                                  color=(255, 255, 0), thickness=1);
            base_imfn=os.path.basename(im_fn);
            img_path = os.path.join(output_dir, base_imfn)
            cv2.imwrite(img_path, ori_im);

    # @staticmethod
    # def the_test(mpath,bbone,prefix,dpath,rpath):
    #     eval=cat_baseline_evaluator(mpath,"0",False,backbone=bbone,prefix=prefix);
    #     eval.eval_on_i15(dpath,rpath,1);
    # @staticmethod
    # def do_test(iter):
    #     prfx="bl";
    #     mp = os.path.join(pathcfg.project_uniabc_data_root, "bl", str(iter));
    #     cat_baseline_evaluator.the_test(mp, "densenet169", prfx, pathcfg.test_dataset,
    #                                   os.path.join(pathcfg.project_uniabc_data_root, "bl" + str(iter)));
    #     tf.reset_default_graph();

class otfLoadout_baseline(abstract_otf_loadout):

    DATAFEEDER = welf_east_batch_generator_mh;
    DATAFEEDERCFG =kot_det_baseline_data_feeder_config;
    MODEL_FN=cat_welf_east_baseline;
    FAMILY_NAME = pathcfg.project_uniabc_data_root;

    def setupmodel_core(this, inputs):
        net =this.MODEL_FN();
        [lc, lb, la] = net.model_training(inputs, this.dev);

        return lc+lb+10*la, [lc, lb, la] ;
    def mkinputs(this):
        inputs = [
            tf.placeholder(shape=[None, None, None, 3], dtype=tf.float32),
            tf.placeholder(shape=[None, None, None, 1], dtype=tf.float32),
            tf.placeholder(shape=[None, None, None, 5], dtype=tf.float32),
            tf.placeholder(shape=[None, None, None, 1], dtype=tf.float32),
            tf.placeholder(shape=[None, None, None, 1], dtype=tf.float32),
        ];

        return inputs;

    def mkfeeddict(this,inputs,batch_data):
        fd={};
        fd[inputs[0]] = batch_data[0];
        fd[inputs[1]] = batch_data[2];
        fd[inputs[2]] = batch_data[3];
        fd[inputs[3]] = batch_data[4];
        fd[inputs[4]] = batch_data[5];
        return fd;


class loadout_mbaseline_oom(otfLoadout_baseline):
    DATAFEEDERCFG = kot_det_baseline_moom_feeder_config;
    FLAG = "mbloom";


class cat_mbaselineoom_evaluator(cat_baseline_evaluator):
    FLAG = "mbloom";
    FEEDER_CFG = kot_det_baseline_moom_feeder_config;


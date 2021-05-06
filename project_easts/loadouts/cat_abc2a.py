from builtins import set

from project_easts.loadouts.baseline import otfLoadout_baseline,cat_baseline_evaluator;
from data_providers.typical_configurations.det_configs import kot_det_baseline_moom_feeder_config;
from project_easts.easts.cat_abc2a import cat_abc2a;
import os;
import cv2;
import numpy as np;

class cat_abc2a_evaluator(cat_baseline_evaluator):
    MODEL_FN=cat_abc2a;
    FLAG="abc2a";
    def DNC_init_set_model(this,backbone):
        this.cls, this.geo, this.sel = this.MODEL_FN().model_testing([this.ipph], backbone);
    def log_etc(this,output_dir,base_imfn,etc):
        sels=(etc[0,:,:,:]*255).astype(np.uint8);
        os.makedirs(os.path.join(output_dir,"sel"),exist_ok=True);
        for i in range(etc.shape[-1]):
            os.makedirs(os.path.join(output_dir,"sel",str(i)),exist_ok=True);
        for i in range (etc.shape[-1]):
            fpath=os.path.join(output_dir,"sel",str(i),base_imfn);
            print(fpath);
            cv2.imwrite(fpath,sels[:,:,:,i]);
    def process(this, image):
        cls_ins, geo_ins, sel_ins = this.sess.run([this.cls, this.geo, this.sel], feed_dict={this.ipph: image});
        bs = this.east_parser.detect_2_boxes([cls_ins, geo_ins], None);
        if ((bs is not None) and (bs.shape[0] > 1)):
            bs = this.east_parser.filter_boxes(bs, cls_ins);
        return bs,sel_ins;

    def process_o_l_rew(this, image, h_limit=1280, v_limit=1280, h_stride=1280, v_stride=1280):
        images, locs, mr, mw = this.split(image, h_limit, v_limit, h_stride, v_stride);
        clss = [];
        geos = [];
        for im in images:
            cls_ins, geo_ins,_ = this.sess.run([this.cls, this.geo], feed_dict={this.ipph: im});
            clss.append(cls_ins.copy());
            geos.append(geo_ins.copy());
        cls = this.merge(clss, locs, mr, mw);
        reg = this.merge(geos, locs, mr, mw);
        return this.parse_rew(cls, reg),None;


class loadout_abc2a(otfLoadout_baseline):
    MODEL_FN=cat_abc2a;
    FLAG="abc2a";

class loadout_abc2aoom(otfLoadout_baseline):
    MODEL_FN=cat_abc2a;
    DATAFEEDERCFG=kot_det_baseline_moom_feeder_config
    FLAG="abc2aoom";
class cat_abc2aoom_evaluator(cat_abc2a_evaluator):
    FLAG="abc2aoom";
    FEEDER_CFG = kot_det_baseline_moom_feeder_config;


class loadout_abc2ar50_oom(otfLoadout_baseline):
    PRET = "pretrained/resnet_v1_50.ckpt";
    BBONE = "resnet_v1_50";
    MODEL_FN=cat_abc2a;
    DATAFEEDERCFG=kot_det_baseline_moom_feeder_config
    FLAG="abc2aoomr50";


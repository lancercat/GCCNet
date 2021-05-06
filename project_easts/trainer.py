
from utils.cat_config import cat_config;

import tensorflow as tf;

import os;

from utils.logger.kot_logger import kot_logger;
from utils.libpath import pathcfg;
from utils.notifier import nekoexpnotification;

from project_easts.loadouts.baseline import loadout_mbaseline_oom;
from project_easts.loadouts.hjb_ham import loadout_hjb_mhamoom;
from project_easts.loadouts.kot_treetail import loadout_kot_mtreetailoom;
from project_easts.loadouts.kot_arc_treetail import loadout_kot_arc_treetail;
from project_easts.loadouts.kot_arc import loadout_kot_marcoom;



from project_easts.loadouts.cat_car import loadout_car;
from project_easts.loadouts.cat_ric import loadout_ricvv;
from project_easts.loadouts.cat_HT import loadout_HT;
from project_easts.loadouts.cat_uni import loadout_uni_kai_om,loadout_uni_kai_oomclp;



USE_CPU=0;
sess_cfg={};
if USE_CPU:
    DEV= "/device:CPU:0";
    sess_cfg["inter_op_parallelism_threads"] = 1;
else:
    DEV = "/device:GPU:0";


def pretrain_data_config():

    config = cat_config();
    config.add_child("onlineicdar", datasource_cfg(
        os.path.join(pathcfg.training_dataset_root,"icdar15-a-13-idx/a.index"),"icdar",5)
                         );
    config.add_child("onlinecoco", datasource_cfg(
        os.path.join(pathcfg.training_dataset_root, "all_tr_data/EAST_data/coco.index"), "icdar", 3)
                     );

    config.add_child("onlinericdar", datasource_cfg(
        os.path.join(pathcfg.training_dataset_root, "all_tr_data/EAST_data/icdar_15_13_0-90.index"), "icdar", 2)
                     );

    config.add_child("onlinesynth", datasource_cfg(
        os.path.join(pathcfg.training_dataset_root, "all_tr_data/EAST_data/synth_cat.index"), "icdar", 1)
                     );

    config.add_child("onlineustid", datasource_cfg(
        os.path.join(pathcfg.training_dataset_root, "all_tr_data/EAST_data/ustid-mk1.index"), "icdar", 1)
                     );

    # config.add_child("onlinemlt", datasource_cfg(
    #     os.path.join(pathcfg.training_dataset_root,/mlt/a.index","mlt",5)
    #                      );

    return config;


def mlt_data_config():

    config = cat_config();
    config.add_child("onlinemlt", datasource_cfg(
        os.path.join(pathcfg.training_dataset_root,"mlt/a.index"),"mlt",5)
                         );

    # config.add_child("onlinemlt", datasource_cfg(
    #     os.path.join(pathcfg.training_dataset_root,/mlt/a.index","mlt",5)
    #                      );

    return config;

def rects_data_config():

    config = cat_config();
    config.add_child("online", datasource_cfg(
        os.path.join(pathcfg.training_dataset_root,"rects/a.index"),"rects",5)
                         );

    return config;

def tencent_rects_data_config():

    config = cat_config();
    config.add_child("online", datasource_cfg(
        os.path.join(pathcfg.training_dataset_root,"rects/a.index"),"icdar",5)
                         );

    return config;


def ln_data_config():

    config = cat_config();
    config.add_child("online", datasource_cfg(
        os.path.join(pathcfg.training_dataset_root,"all_line_tr_data/td500_train.index"),"icdar",5)
                         );

    config.add_child("online", datasource_cfg(
        os.path.join(pathcfg.training_dataset_root,"all_line_tr_data/rctw.index"),"icdar",3)
                         );
    config.add_child("online", datasource_cfg(
        os.path.join(pathcfg.training_dataset_root,"all_line_tr_data/sv1k_training.index"),"icdar",2)
                         );

    # config.add_child("onlinemlt", datasource_cfg(
    #     os.path.join(pathcfg.training_dataset_root,/mlt/a.index","mlt",5)
    #                      );

    return config;



def det_data_config():

    config = cat_config();
    config.add_child("onlineicdar", datasource_cfg(
        os.path.join(pathcfg.training_dataset_root,"icdar15-a-13-idx/a.index"),"icdar",5)
                         );

    # config.add_child("onlinemlt", datasource_cfg(
    #     os.path.join(pathcfg.training_dataset_root,/mlt/a.index","mlt",5)
    #                      );

    return config;




def datasource_cfg(path, type, importance,has_gt=1):
    config = cat_config();
    config.set("index", path);
    config.set("type", type);
    config.set("importance", importance);
    config.set("has_gt",has_gt);
    return config;

def pretrain(train_fn,total_iter=9999*41):
    dataset_cfgs = pretrain_data_config();
    handle = train_fn();
    handle.train_setup(dataset_cfgs, total_iter=total_iter);
    handle.train(0);

def train(train_fn,total_iter=9999*81):
    dataset_cfgs = det_data_config();
    handle=train_fn();
    handle.train_setup(dataset_cfgs,total_iter=total_iter)
    handle.train(0);

def train_cont(train_fn):
    dataset_cfgs = det_data_config();
    handle=train_fn();
    handle.train_setup(dataset_cfgs,total_iter=9999*81)
    handle.train(409959);
def kickrestoreMLT(train_fn,from_iter,total_iter=9999*81):
    dataset_cfgs = mlt_data_config();
    handle = train_fn();
    tpath = os.path.join(pathcfg.project_uniabc_data_root, "MLT");
    try:
        os.mkdir(tpath);
    except:
        pass;
    handle.train_setup(dataset_cfgs, tpath=tpath, total_iter=total_iter)
    handle.train(0,tpretrained=from_iter);

def trainMLT(train_fn,total_iter=9999*81):
    dataset_cfgs = mlt_data_config();
    handle=train_fn();
    tpath = os.path.join(pathcfg.project_uniabc_data_root, "MLT");
    try:
        os.mkdir(tpath);
    except:
        pass;
    handle.train_setup(dataset_cfgs,tpath=tpath,total_iter=total_iter)
    handle.train(0);

def trainLINE(train_fn,total_iter=9999*81):
    dataset_cfgs = ln_data_config();
    handle=train_fn();
    tpath = os.path.join(pathcfg.project_uniabc_data_root, "LINE");
    try:
        os.mkdir(tpath);
    except:
        pass;
    handle.train_setup(dataset_cfgs,tpath=tpath,total_iter=total_iter)
    handle.train(0);
def trainrects(train_fn,total_iter=9999*81):
    dataset_cfgs = rects_data_config();
    handle=train_fn();
    tpath = os.path.join(pathcfg.project_uniabc_data_root, "ReCTS");
    try:
        os.makedirs(tpath,exist_ok=True);
    except:
        pass;
    handle.train_setup(dataset_cfgs,tpath=tpath,total_iter=total_iter)
    handle.train(0);
def train_tencent_rects(train_fn,total_iter=9999*25):
    dataset_cfgs = tencent_rects_data_config()
    handle = train_fn();
    tpath = os.path.join(pathcfg.project_uniabc_data_root, "tencent_rects");
    try:
        os.makedirs(tpath, exist_ok=True);
    except:
        pass;
    handle.train_setup(dataset_cfgs, tpath=tpath, total_iter=total_iter)
    handle.train(0);
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = os.sys.argv[1];
    kot_logger.init(suffix=".jpg");
    # train_uniabc(0,DEV);
    # train_abcR90(0,DEV)
    # train(loadout_car);
    # train(loadout_ricvv);
    # train(loadout_hdabc);
    train_tencent_rects(loadout_mbaseline_oom);



    # train(loadout_uni);
    # train_baseline_om(0,DEV)
    # train(0,DEV);
    # train_baseline_ol(0,DEV)

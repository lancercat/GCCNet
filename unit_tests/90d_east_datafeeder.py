from data_providers.data_generators.uniform_batch.welf_east_data_genrator import cat_90d_east_batch_generator_mh;
from utils.cat_config import cat_config;
from  utils.libpath import pathcfg;
import os;

import time;
_cfg={};
_cfg["batch_size"]=1;
_cfg["vis"]=True;

def datasource_cfg(path, type, importance):
    config = cat_config();
    config.set("index", path);
    config.set("type", type);
    config.set("importance", importance);
    return config;

def data_config():
    config = cat_config();

    config.add_child("onlinerctw", datasource_cfg(
        os.path.join(pathcfg.training_dataset_root,"rctw/a.index"),"rctw",1)
                         );

    config.add_child("onlinemlt", datasource_cfg(
        os.path.join(pathcfg.training_dataset_root,"mlt/a.index"),"mlt",1)
                         );
    return config;
cfg=cat_90d_east_batch_generator_mh.get_default_config(data_config(),batch_size=1,vis=1);
dg=cat_90d_east_batch_generator_mh(cfg);

while True:
    t=time.time();
    for i in range(10):
        dg.next_batch();
        print("done");
    print("here"+str(time.time()-t));



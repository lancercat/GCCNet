from data_providers.data_generators.uniform_batch.welf_east_data_genrator import welf_east_batch_generator_mh;
from utils.cat_config import cat_config;
from data_providers.typical_configurations.det_configs import kot_det_baseline_data_feeder_config;

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

    config.add_child("rects", datasource_cfg(
        os.path.join(pathcfg.training_dataset_root,"rects/a.index"),"rects",1)
                         );

    return config;
DATAFEEDER=welf_east_batch_generator_mh;

cfg=DATAFEEDER.get_default_config(data_config(),**(kot_det_baseline_data_feeder_config().configs()));
dg=DATAFEEDER(cfg);

while True:
    t=time.time();
    for i in range(10):
        res=dg.next_batch();
        for j in res:
            print(j);
        print("done");
    print("here"+str(time.time()-t));



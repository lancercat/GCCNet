from utils.libpath import pathcfg
from utils.cat_config import cat_config;
from data_providers.data_generators.uniform_batch.welf_east_data_genrator import welf_east_batch_generator_mh;
import os;

def datasource_cfg(path, type, importance,has_gt=1):
    config = cat_config();
    config.set("index", path);
    config.set("type", type);
    config.set("importance", importance);
    config.set("has_gt",has_gt);
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

cfg=welf_east_batch_generator_mh.get_default_config(mlt_data_config(),batch_size=1);

g=welf_east_batch_generator_mh(cfg);
import time;
while True:
    start=time.time();
    for i in range(20):
        g.next_batch();

    print ("here");
    print(time.time()-start);
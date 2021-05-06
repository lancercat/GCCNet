from utils.cat_config import cat_config;
from data_providers.givers.determined_size.std_east_givers import welf_east_giver;
from data_providers.data_generators.uniform_batch.kot_abstract_det_batch_generator import kot_abstract_det_batch_generator;

# provide batch level control. Tells giver the desired format of EACH item in a batch
class welf_east_batch_generator_mh(kot_abstract_det_batch_generator):
    GIVER_FN=welf_east_giver;
    @classmethod
    def get_default_config(cls,dataset_cfgs,**args):
        config = cat_config();
        config.set_list("batch_sizes", args["batch_sizes"]);
        config.set_list("sizes",args["sizes"]);

        config.set("required_cnt", 6);
        iconf=cat_config();
        iconf.set("shuffle_flag", args.get("shuffle_flag", True));
        config.add_child("iterator_config",iconf);
        gconf=cls.GIVER_FN.get_default_config(**args);
        config.add_child("giver_cfg",gconf);
        config.add_child("data_sources", dataset_cfgs);
        return config;

    def init_callback(this,config):
        pass;

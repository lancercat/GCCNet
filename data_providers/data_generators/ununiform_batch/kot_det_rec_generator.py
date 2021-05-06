from data_providers.givers.undetermined_size.kot_rec_det_giver import kot_fots_giver;
from data_providers.iterators.cat_e2e_dataset_iterator import cat_e2e_dataset_iterator;
from data_providers.kot_abstract_data_generator import kot_abstract_data_generator;

from utils.cat_config import cat_config;

class kot_det_rec_generator_mh(kot_abstract_data_generator):
    GIVER_FN=kot_fots_giver;
    ITERATOR_FN = cat_e2e_dataset_iterator;

    @classmethod
    def get_default_config(cls,dataset_cfgs,**args):
        config = cat_config();
        # A "Dummy" value.
        config.set_list("sizes", [9]);

        config.set_list("batch_sizes",[3]);
        config.set("required_cnt", 4+5+4);
        iconf=cat_config();
        iconf.set_item_wdef(args,"shuffle_flag", True);
        dp=dataset_cfgs.get(str,"dict_path");
        iconf.set("dict_path",dp);
        config.add_child("iterator_config",iconf);
        gconf=cls.GIVER_FN.get_default_config(dict_path=dp,**args);
        config.add_child("giver_cfg",gconf);
        config.add_child("data_sources", dataset_cfgs);
        return config;


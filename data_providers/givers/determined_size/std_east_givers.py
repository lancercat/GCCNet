from utils.cat_config import cat_config;
from data_providers.module_anno.east_anno import kot_libeast
from data_providers.givers.determined_size.kot_abstract_det_giver import kot_abstract_det_giver;
class welf_east_giver(kot_abstract_det_giver):

    def det_init_callback(this,config):
        this.generators = [kot_libeast(config.get_child("anno_config"))];

    @classmethod
    def get_default_config(cls,**args):
        config = cat_config();

        anno_config = cat_config();
        anno_config.set("margin_dc", args.get("margin_dc", 0));
        anno_config.set("min_text_size", args.get("min_text_size", 5));
        config.add_child("anno_config",anno_config);


        config.set_list("random_scale", args.get("random_scale", [0.5,0.8, 1, 1.5, 2]));
        config.set("background_ratio", args.get("background_ratio", 0));
        config.set("geometry", args.get("geometry", "RBOX"));
        config.set("vis", args.get("vis", False));
        config.set_list("mean", args.get("mean", [127, 127, 127]));
        config.set_list("var", args.get("var", [127, 127, 127]));
        config.set("featuremap_stride",args.get("featuremap_stride",4));
        config.set_list("rotate_ranges_max", args.get("rotate_ranges_max", []));
        config.set_list("rotate_ranges_min", args.get("rotate_ranges_min", []));
        config.set_list("rotate_ranges_freq", args.get("rotate_ranges_freq", []));

        return config;
    def nec_cbk(this,raw_entry,task_item):
        task_item.append(raw_entry.im_path);
        pass;


class hjb_h2v1_east_giver(welf_east_giver):
    @classmethod
    def get_default_config(cls,**args):
        config = cat_config();
        config.set("margin_dc", args.get("margin_dc", 0));
        config.set("min_text_size", args.get("min_text_size", 5));
        config.set_list("random_scale", args.get("random_scale", [0.5, 0.8, 1, 1.5, 2]));
        config.set("background_ratio", args.get("background_ratio", 0));
        config.set("geometry", args.get("geometry", "RBOX"));
        config.set("vis", args.get("vis", False));
        config.set_list("mean", args.get("mean", [127, 127, 127]));
        config.set_list("var", args.get("var", [127, 127, 127]));
        config.set("featuremap_stride", args.get("featuremap_stride", 4));
        config.set_list("rotate_ranges_max", args.get("rotate_ranges_max", [100,10]));
        config.set_list("rotate_ranges_min", args.get("rotate_ranges_min", [80,-10]));
        config.set_list("rotate_ranges_freq", args.get("rotate_ranges_freq", [1,2]));

        return config;


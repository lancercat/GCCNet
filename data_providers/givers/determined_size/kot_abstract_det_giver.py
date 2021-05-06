from data_providers.givers.kot_abstract_giver import kot_abstract_giver;
import numpy as np;
import random;

from data_providers.augumenter.augumenters import augumenter;

class kot_abstract_det_giver(kot_abstract_giver):


    def det_init_callback(this,config):
        pass;

    def init_call_back(this,config):
        this.random_scale = config.get_list(float, "random_scale");
        this.background_ratio = config.get(float, "background_ratio");
        this.geometry = config.get(str, "geometry");
        this.vis = config.get(int, "vis");
        this.mean = np.array(config.get_list(float, "mean"));
        this.var = np.array(config.get_list(float, "var"));
        this.featuremap_stride = config.get(int, "featuremap_stride");

        maxs=config.get_list(int, "rotate_ranges_max");
        mins=config.get_list(int, "rotate_ranges_min");
        freqs=config.get_list(int, "rotate_ranges_freq");
        this.rac=[];
        for i in range(len(maxs)):
            for j in range(freqs[i]):
                this.rac.append([mins[i],maxs[i]]);

        this.det_init_callback(config);

    def do_augumentation(this, raw_entry, size):
        rr =None;
        if len(this.rac):
            rr=random.choice(this.rac);

        im, text_polys, text_tags = augumenter.get_east_augumented(raw_entry, input_size=size, background_ratio=this.background_ratio,
                                                random_scale=this.random_scale, mean=this.mean, var=this.var, rotate_range=rr);
        return im, text_polys, text_tags;
    def nec_cbk(this,raw_entry,task_item):

        pass;

    def  next_entry_core(this,required,raw_entry,size,bid):

        im, text_polys, text_tags = this.do_augumentation(raw_entry, size);

        if (im is None):
            return 0;
        new_h, new_w, _ = im.shape

        task_items = [im[:, :, ::-1].astype(np.float32)];
        this.nec_cbk(raw_entry,task_items);


        try:
            for gid in range(len(this.generators)):
                tr = this.generators[gid].generate_gt((new_h, new_w), text_polys, text_tags, this.featuremap_stride);
                for i in tr:
                    if i is None:
                        return 0;
                    task_items.append(i);
        except:
            print("Boooooomed");
            return 0;

        for i in range(len(task_items)):
            required[i].append(task_items[i]);
            # print("got")
        return 1;

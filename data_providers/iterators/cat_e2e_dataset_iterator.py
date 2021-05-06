
import numpy as np;
from data_providers.dataset import gt_desc_factory;
import os;
import random;

class cat_e2e_dataset_iterator:
    def DNC_init(this, config, dsc):
        this.iter = 0;
        # load your data here
        this.entries = [];
        this.shuffle_flag = config.get(int, "shuffle_flag");
        this.dsc = dsc
        this.DNC_load_data()
        this.shuffle(this.index);

    def DNC_load_data(this):
        if(len(this.entries)):
            del this.entries;
            del this.item_cnt;
        this.entries = [];
        this.entries += gt_desc_factory.build_all_with_index(this.dsc.get(str,"index"), this.dsc.get(str,"type"));

        this.item_cnt = len(this.entries);
        this.index = np.arange(0, this.item_cnt);
    def shuffle(this,index):
        this.re.shuffle(index);

    def __init__(this, config, dsc):
        this.seed=os.getpid()%0xca71;
        this.re=random.Random(this.seed);
        this.DNC_init(config,dsc);
        
        # something else;

        pass;
    def next_entry_id(this):
        if (this.iter == this.item_cnt):
            this.iter = 0;
            print("shuffling " + str(this.item_cnt) + " images")
            this.shuffle(this.index);
        elif (this.iter > this.item_cnt):
            exit(9);
        i = this.index[this.iter];
        this.iter += 1;
        return i

    def next_entry(this):
        raw_entry=this.entries[this.next_entry_id()];
        return raw_entry;
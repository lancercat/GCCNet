import random
from utils.prof_utils import get_size;

class kot_abstract_data_generator:

    ITERATOR_FN=None;
    GIVER_FN=None;

    def init_callback(this,config):
        pass;
    def init_set_givers(this,config):
        dscc = config.get_child("data_sources");
        for dsc in dscc.children_index:
            dsc = dscc.get_child(dsc);
            I=this.ITERATOR_FN(config.get_child("iterator_config"), dsc);
            G=this.GIVER_FN(config.get_child("giver_cfg"),I);
            this.givers.append(G);
            for i in range(dsc.get(int, "importance")):
                this.giver_index.append(len(this.givers) - 1);

    def __init__(this,config):
        this.givers = [];
        this.giver_index = [];
        this.required_cnt = config.get(int, "required_cnt")
        this.init_set_givers(config);

        this.batch_sizes=config.get_list(int,"batch_sizes");
        this.sizes=config.get_list(int,"sizes");
        this.keys=range(len(this.batch_sizes));

    def try_get_a_batch(this,giver,required,size,bid):
        return giver.next_entry(required, size, bid);

    def next_batch(this):
        required=[];
        scid=random.choice(this.keys);
        batch_size=this.batch_sizes[scid];
        size=this.sizes[scid];
        #print(batch_size,size);
        for i in range(this.required_cnt):
            required.append([]);
        has = 0;
        while has < batch_size:
            giver_id = random.choice(this.giver_index);
            giver = this.givers[giver_id];
            has += this.try_get_a_batch(giver,required,size,has);
        return required;

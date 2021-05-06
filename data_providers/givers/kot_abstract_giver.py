import numpy as np;
# u ever need sync, sync on giver level.

class kot_abstract_giver:
    def init_call_back(this,config):
        pass;

    def __init__(this,config,iterator):
        this.iterator = iterator;
        this.generators = [];

        this.init_call_back(config);


    def next_entry_core(this,required,raw_entry,size,bid):
        pass;

    def next_entry(this,required,size,bid):
        raw_entry = this.iterator.next_entry();
        return this.next_entry_core(required,raw_entry,size,bid);


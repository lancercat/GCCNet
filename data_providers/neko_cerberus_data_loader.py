# A multi-thread data loader designed for different task training at same time,
#  saves you all the trouble joining threads.
import numpy as np;

from tensorflow.keras.utils import GeneratorEnqueuer;

import time;
class neko_naive_generator:
    @staticmethod
    def __generate(generator_functor, generator_cfg):
        generator = generator_functor(generator_cfg);
        while True:
            yield generator.next_batch();

    def __init__(this, generator_functor, generator_cfg, num_workers):
        this.num_workers = num_workers;
        sequence=this.__generate(generator_functor, generator_cfg);
        this.enqueuer = GeneratorEnqueuer(sequence, use_multiprocessing=True);

    def start(this):
        this.enqueuer.start(max_queue_size=24, workers=this.num_workers);

    def stop(this):
        if this.enqueuer is not None:
            this.enqueuer.stop();

    def get(this):
        generator_output = None;
        while this.enqueuer.is_running():
            if not this.enqueuer.queue.empty():
                generator_output = this.enqueuer.queue.get();
                break
            else:
                time.sleep(0.01);
        return generator_output[1];

# well Cerberus is the most fit one here,
# tho I am NOT a dog cat.
class neko_cerberus_generator:
    @classmethod
    def get_default_config(cls,**kwargs):
        config = {};
        config["num_workers"] = kwargs.get("num_workers",9);
        return config;

    @staticmethod
    def __generate(generator_functors, generator_cfgs):
        generators = [];
        for i in range(len(generator_functors)):
            generators.append(generator_functors[i](generator_cfgs[i]));
        while True:
            ret = [];
            for g in generators:
                ret.append(g.next_batch());
            yield ret;

    def start(this):
        sequence=this.__generate(this.generator_functors,this.generator_cfgs);

        this.enqueuer = GeneratorEnqueuer(sequence, use_multiprocessing=True)
        this.enqueuer.start(max_queue_size=24, workers=this.num_workers)

    def stop(this):
        if this.enqueuer is not None:
            this.enqueuer.stop()

    def add_task(this, functor, config):
        this.generator_functors.append(functor);
        this.generator_cfgs.append(config);

    def __init__(this, config):
        this.generator_functors = [];
        this.generator_cfgs = [];

        this.num_workers = config["num_workers"];

    def get(this):
        generator_output = None;
        # while this.enqueuer.is_running():
        #     if not this.enqueuer.queue.empty():
        #         baz=this.enqueuer.queue.get();
        #         [succ,generator_output] = baz;
        #         if(succ):
        #             break
        #     else:
        #         time.sleep(0.01)
        # force yielding
        return this.enqueuer.get().__next__();


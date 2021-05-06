import tensorflow as tf;
from tensorflow.contrib import slim;
import os;
import numpy as np;
import cv2,time;

from data_providers.neko_cerberus_data_loader import neko_cerberus_generator;
from utils.transplanter import load_pretrained;
from utils.libpath import pathcfg;
from utils.logger.kot_logger import kot_logger;
from abc import abstractmethod
from  generic_evaluators.cat_det_evaluator import cat_abstract_detection_evaluator;

from utils.notifier import nekoexpnotification;
from generic.abstract_loadout import abstract_loadout;

class abstract_otf_loadout(abstract_loadout):
    PRET="pretrained/densenet_169/tf-densenet169.ckpt";

    @abstractmethod
    def mkinputs(this):
        pass;

    @abstractmethod
    def mkfeeddict(this, inputs, batch_data):
        pass;

    @abstractmethod
    def setupmodel_core(this, inputs):
        pass;


    #################Before Runtime##########################
    # TODO[--] the gpus should be a list
    def train_setup_callback(this, use_gpu=1):
        sess_cfg = {};
        if not use_gpu:
            dev = "/device:CPU:0";
            sess_cfg["inter_op_parallelism_threads"] = 1;
        else:
            dev = "/device:GPU:0";
        this.dev = dev;
        this.sess_cfg = sess_cfg;
        this.sess = tf.Session(config=tf.ConfigProto(**this.sess_cfg));

    def setupmodel(this, model_args):
        this.inputs=this.mkinputs();
        this.loss, this.log_tar = this.setupmodel_core(this.inputs);

    def set_up_saver(this):
        this.v = slim.get_variables(this.PRFX, None, tf.GraphKeys.GLOBAL_VARIABLES);
        this.saver = tf.train.Saver(var_list=this.v, max_to_keep=80);


    ################Run time before loop###############

    def set_up_optimizer(this, restore_from):
        learning_rate = this.BASE_LR;
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(restore_from),
                                      trainable=False);
        lr = tf.train.exponential_decay(learning_rate, global_step, decay_steps=this.DECAY_EACH,
                                        decay_rate=this.DECAY_RATIO,
                                        staircase=True);
        optimizer = tf.train.AdamOptimizer(learning_rate=lr);
        grads = optimizer.compute_gradients(this.loss + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES));
        variable_averages = tf.train.ExponentialMovingAverage(this.moving_average_decay, global_step);

        variables_averages_op = variable_averages.apply(tf.trainable_variables());
        if (this.GCT > 0):
            grads = [(tf.clip_by_value(grad, -this.GCT, this.GCT), var) for grad, var in grads]
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step);
        batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))

        with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
            this.train_op = tf.no_op(name='train_op');

    def load_pretrained(this, tpretrained):
        this.sess.run(tf.global_variables_initializer());
        pret = None;
        if (tpretrained is None):
            pret = os.path.join(pathcfg.generic_data_root, this.PRET);
            load_pretrained(this.v, this.PRFX, this.sess, pret
                            );
        else:
            this.saver.restore(this.sess, os.path.join(this.fpath, this.PRFX, str(tpretrained)));

    def restore(this,restore_from):
        if (restore_from):
            this.saver.restore(this.sess, os.path.join(this.fpath, this.PRFX, str(restore_from)));

    ################Run time within loop##########################

    def save_model(this,iter,loss):
        this.saver.save(this.sess, os.path.join(this.tpath, this.FLAG, str(iter)));


    def step(this,iter_id):
        bd = this.feeder.get()[0];
        fd = this.mkfeeddict(this.inputs, bd);

        # print("before_iter, memfootprint", current_mem_usage());
        # print("after_iter, memfootprint", current_mem_usage());
        loss=None;

        if (iter_id % this.log_each == 0):
            ret = this.sess.run([this.loss] + this.log_tar + [this.train_op],
                                feed_dict=fd);
            print("Iter", str(iter_id), "loss is ", str(ret[0]));
            print("Details");
            for j in range(1, len(ret) - 1):
                print(ret[j]);
            loss=ret[0];
        else:
            _ = this.sess.run([this.train_op],
                              feed_dict=fd);
        return loss;





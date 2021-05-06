#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import division
import tensorflow as tf

from slime.nets import nets_factory
from slime.feature_fuser.upsampler import welf_slime_unpool;
from slime.modules.east_parser import welf_slime_east_parser;
from slime.rollouts.abstract_rollouts import abstract_rollout


class welf_slime_east(abstract_rollout):
    def __init__(this, config):
        this.network_name = config.get(str, "network");
        this.weight_decay=config.get(float,"weight_decay");
        this.vscope=config.get(str,"scope");
        this.text_scale=512;



    def model(this, inputs, is_training):
        '''
        define the model, we use slim's implemention of resnet
        '''
        with tf.variable_scope(this.vscope):
            base_network = nets_factory.get_network_fn(this.network_name, num_classes=2, weight_decay=this.weight_decay,
                                                       is_training=is_training)
            logits, end_points = base_network(inputs[0]);
            f = [end_points['pool5'], end_points['pool4'],
                 end_points['pool3'], end_points['pool2']];
            values = None ;#end_points.value;

            # with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
            #    logits, end_points = resnet_v1.resnet_v1_50(images, is_training=is_training, scope='resnet_v1_50')
            g = welf_slime_unpool().model(f, is_training, values=values, weight_decay=this.weight_decay);
            return welf_slime_east_parser().model(g, is_training, values=values, weight_decay=this.weight_decay,text_scale=this.text_scale);


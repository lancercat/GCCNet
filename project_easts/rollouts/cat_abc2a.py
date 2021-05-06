from slime.rollouts.libeast import welf_slime_east;
from slime.nets import nets_factory;
from slime.feature_fuser.abc_upsampler import cat_abca_unpool;
from slime.modules.east_parser_abc import cat_abca_slime_east_parser,cat_abcah_slime_east_parser;

import tensorflow as tf;

class cat_abc2a_east(welf_slime_east):
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
            values = None;  # end_points.value;

            # with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
            #    logits, end_points = resnet_v1.resnet_v1_50(images, is_training=is_training, scope='resnet_v1_50')
            g = cat_abca_unpool().model(f, is_training, values=values, weight_decay=this.weight_decay);
            return cat_abca_slime_east_parser().model(g, is_training, values=values, weight_decay=this.weight_decay,
                                                  text_scale=this.text_scale);
class cat_abc2ah_east(welf_slime_east):
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
            values = None;  # end_points.value;

            # with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
            #    logits, end_points = resnet_v1.resnet_v1_50(images, is_training=is_training, scope='resnet_v1_50')
            g = cat_abca_unpool().model(f, is_training, values=values, weight_decay=this.weight_decay);
            return cat_abcah_slime_east_parser().model(g, is_training, values=values, weight_decay=this.weight_decay,
                                                  text_scale=this.text_scale);

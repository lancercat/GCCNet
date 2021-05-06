from tensorflow.contrib import slim
import tensorflow as tf;
import numpy as np;

class welf_slime_east_parser:
    def model(this,inputs,is_training,**kwargs):
        # result parsing should not be bounded to feature fusion
        with tf.variable_scope('result_parser', values=kwargs["values"]):
            batch_norm_params = {
                'decay': 0.997,
                'epsilon': 1e-5,
                'scale': True,
                'is_training': is_training
            }
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                weights_regularizer=slim.l2_regularizer(kwargs["weight_decay"])):

                # here we use a slightly different way for regression part,
                # we first use a sigmoid to limit the regression range, and also
                # this is do with the angle map
                F_score = slim.conv2d(inputs[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                # 4 channel of axis aligned bbox and 1 channel rotation angle
                geo_map = slim.conv2d(inputs[3], 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * kwargs["text_scale"];
                angle_map = (slim.conv2d(inputs[3], 1, 1, activation_fn=tf.nn.sigmoid,
                                         normalizer_fn=None) - 0.5) * np.pi / 2  # angle is between [-45, 45]
                F_geometry = tf.concat([geo_map, angle_map], axis=-1)

        return  F_score,F_geometry;

class welf_slime_east_parser_mt:
    def model(this,inputs,is_training,**kwargs):
        # result parsing should not be bounded to feature fusion
        with tf.variable_scope('result_parser', values=kwargs["values"]):
            batch_norm_params = {
                'decay': 0.997,
                'epsilon': 1e-5,
                'scale': True,
                'is_training': is_training
            }
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                weights_regularizer=slim.l2_regularizer(kwargs["weight_decay"])):

                # here we use a slightly different way for regression part,
                # we first use a sigmoid to limit the regression range, and also
                # this is do with the angle map
                F_score = slim.conv2d(inputs[0][3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                # 4 channel of axis aligned bbox and 1 channel rotation angle
                geo_map = slim.conv2d(inputs[1][3], 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * kwargs["text_scale"];
                angle_map = (slim.conv2d(inputs[2][3], 1, 1, activation_fn=tf.nn.sigmoid,
                                         normalizer_fn=None) - 0.5) * np.pi / 2  # angle is between [-45, 45]
                F_geometry = tf.concat([geo_map, angle_map], axis=-1)

        return  F_score,F_geometry;

class welf_slime_east_cls_parser:
    def model(this,inputs,is_training,**kwargs):
        # result parsing should not be bounded to feature fusion
        with tf.variable_scope('east_cls', values=kwargs["values"]):
            batch_norm_params = {
                'decay': 0.997,
                'epsilon': 1e-5,
                'scale': True,
                'is_training': is_training
            }
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                weights_regularizer=slim.l2_regularizer(kwargs["weight_decay"])):

                # here we use a slightly different way for regression part,
                # we first use a sigmoid to limit the regression range, and also
                # this is do with the angle map
                F_score = slim.conv2d(inputs[0], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                # 4 channel of axis aligned bbox and 1 channel rotation angle
        return  F_score;


class welf_slime_east_bbox_parser:
    def model(this,inputs,is_training,**kwargs):
        # result parsing should not be bounded to feature fusion
        with tf.variable_scope('east_bbox', values=kwargs["values"]):
            batch_norm_params = {
                'decay': 0.997,
                'epsilon': 1e-5,
                'scale': True,
                'is_training': is_training
            }
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                weights_regularizer=slim.l2_regularizer(kwargs["weight_decay"])):

                geo_map = slim.conv2d(inputs[0], 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * kwargs["text_scale"];

        return  geo_map;


class welf_slime_east_angle_parser:
    def model(this,inputs,is_training,**kwargs):
        # result parsing should not be bounded to feature fusion
        with tf.variable_scope('east_angle', values=kwargs["values"]):
            batch_norm_params = {
                'decay': 0.997,
                'epsilon': 1e-5,
                'scale': True,
                'is_training': is_training
            }
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                weights_regularizer=slim.l2_regularizer(kwargs["weight_decay"])):
                angle_map = (slim.conv2d(inputs[0], 1, 1, activation_fn=tf.nn.sigmoid,
                                         normalizer_fn=None) - 0.5) * np.pi / 2  # angle is between [-45, 45]
        return  angle_map;


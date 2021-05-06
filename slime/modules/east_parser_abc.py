from tensorflow.contrib import slim
import tensorflow as tf;
import numpy as np;
import math;
from slime.modules.neko_abcs import neko_abc_collection

class cat_abc_slime_east_parser:
    @staticmethod
    def cat_abc(feature, cnt_b, BS):
        selector = slim.conv2d(feature, cnt_b, 1, activation_fn=None, normalizer_fn=None)
        selector = tf.nn.softmax(selector, axis=-1);
        selector = tf.expand_dims(selector, -2);

        to_select = slim.conv2d(feature, cnt_b * BS, 1);
        shp = tf.shape(to_select);
        to_select = tf.expand_dims(to_select, -2);
        to_select = tf.reshape(to_select, [shp[0], shp[1], shp[2], BS, cnt_b]);

        print(selector.shape);
        selected = to_select * selector;
        return selector, tf.reduce_sum(selected, axis=-1);
    def model(this,inputs,is_training,**kwargs):
        # result parsing should not be bounded to feature fusion
        with tf.variable_scope('result_parser_ABC', values=kwargs["values"]):
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
                sel, ff = this.cat_abc(inputs[3], 9, inputs[3].shape[-1]);

                F_score = slim.conv2d(ff, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                # 4 channel of axis aligned bbox and 1 channel rotation angle
                geo_map = slim.conv2d(ff, 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * kwargs["text_scale"];
                angle_map = (slim.conv2d(ff, 1, 1, activation_fn=tf.nn.sigmoid,
                                         normalizer_fn=None) - 0.5) * np.pi / 2  # angle is between [-45, 45]
                F_geometry = tf.concat([geo_map, angle_map], axis=-1)

        return  F_score,F_geometry,sel;

class cat_abc_nss_slime_east_parser:
    @staticmethod
    def cat_abc(feature, cnt_b, BS):
        selector = slim.conv2d(feature, cnt_b, 1, activation_fn=None, normalizer_fn=None)
        selector = tf.nn.softmax(selector, axis=-1);
        selector = tf.expand_dims(selector, -2);

        to_select = slim.conv2d(feature, cnt_b * BS, 1);
        shp = tf.shape(to_select);
        to_select = tf.expand_dims(to_select, -2);
        to_select = tf.reshape(to_select, [shp[0], shp[1], shp[2], BS, cnt_b]);

        print(selector.shape);
        selected = to_select * selector;
        return selector, tf.reduce_sum(selected, axis=-1);
    def model(this,inputs,is_training,**kwargs):
        # result parsing should not be bounded to feature fusion
        with tf.variable_scope('result_parser_ABC', values=kwargs["values"]):
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
                selc, ffc = this.cat_abc(inputs[3], 3, inputs[3].shape[-1]);

                selr, ffr = this.cat_abc(inputs[3], 5, inputs[3].shape[-1]);

                sela, ffa = this.cat_abc(inputs[3], 3, inputs[3].shape[-1]);

                F_score = slim.conv2d(ffc, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                # 4 channel of axis aligned bbox and 1 channel rotation angle
                geo_map = slim.conv2d(ffr, 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * kwargs["text_scale"];
                angle_map = (slim.conv2d(ffa, 1, 1, activation_fn=tf.nn.sigmoid,
                                         normalizer_fn=None) - 0.5) * np.pi / 2  # angle is between [-45, 45]
                F_geometry = tf.concat([geo_map, angle_map], axis=-1)

        return  F_score,F_geometry,tf.concat([selc,selr,sela],axis=-1);
class cat_abcs_slime_east_parser:
    # Anchor function has more non-linearity

    def model(this,inputs,is_training,**kwargs):
        # result parsing should not be bounded to feature fusion
        with tf.variable_scope('result_parser_ABC', values=kwargs["values"]):
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
                with tf.variable_scope('cls', values=kwargs["values"]):
                    selc, ffc = neko_abc_collection.cat_coabcs(inputs[3],inputs[3], 3, inputs[3].shape[-1],kwargs["weight_decay"],16);

                with tf.variable_scope('reg', values=kwargs["values"]):
                    selr, ffr = neko_abc_collection.cat_coabcs(inputs[3],inputs[3], 5, inputs[3].shape[-1], kwargs["weight_decay"],32);

                with tf.variable_scope('ang', values=kwargs["values"]):
                    sela, ffa = neko_abc_collection.cat_coabcs(inputs[3],inputs[3], 3, inputs[3].shape[-1],kwargs["weight_decay"],16);

                F_score = slim.conv2d(ffc, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                # 4 channel of axis aligned bbox and 1 channel rotation angle
                geo_map = slim.conv2d(ffr, 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * kwargs["text_scale"];
                angle_map = (slim.conv2d(ffa, 1, 1, activation_fn=tf.nn.sigmoid,
                                         normalizer_fn=None) - 0.5) * np.pi / 2  # angle is between [-45, 45]
                F_geometry = tf.concat([geo_map, angle_map], axis=-1)

        return  F_score,F_geometry,tf.concat([selc,selr,sela],axis=-1);

class cat_abca_slime_east_parser:
    #normalized as attention is all you need
    def model(this,inputs,is_training,**kwargs):
        # result parsing should not be bounded to feature fusion
        with tf.variable_scope('result_parser_ABC', values=kwargs["values"]):
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
                with tf.variable_scope('cls', values=kwargs["values"]):
                    selc, ffc = neko_abc_collection.cat_coabca(inputs[3],inputs[3], 3, inputs[3].shape[-1],kwargs["weight_decay"]);

                with tf.variable_scope('reg', values=kwargs["values"]):
                    selr, ffr = neko_abc_collection.cat_coabca(inputs[3],inputs[3], 5, inputs[3].shape[-1], kwargs["weight_decay"]);

                with tf.variable_scope('ang', values=kwargs["values"]):
                    sela, ffa = neko_abc_collection.cat_coabca(inputs[3],inputs[3], 3, inputs[3].shape[-1],kwargs["weight_decay"]);

                F_score = slim.conv2d(ffc, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                # 4 channel of axis aligned bbox and 1 channel rotation angle
                geo_map = slim.conv2d(ffr, 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * kwargs["text_scale"];
                angle_map = (slim.conv2d(ffa, 1, 1, activation_fn=tf.nn.sigmoid,
                                         normalizer_fn=None) - 0.5) * np.pi / 2  # angle is between [-45, 45]
                F_geometry = tf.concat([geo_map, angle_map], axis=-1)

        return  F_score,F_geometry,tf.concat([selc,selr,sela],axis=-1);

class cat_abcn_slime_east_parser:
    #normalized as attention is all you need
    def model(this,inputs,is_training,**kwargs):
        # result parsing should not be bounded to feature fusion
        with tf.variable_scope('result_parser_ABC', values=kwargs["values"]):
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
                with tf.variable_scope('cls', values=kwargs["values"]):
                    selc, ffc = neko_abc_collection.cat_coabcn(inputs[3],inputs[3], 3, inputs[3].shape[-1],kwargs["weight_decay"]);

                with tf.variable_scope('reg', values=kwargs["values"]):
                    selr, ffr = neko_abc_collection.cat_coabcn(inputs[3],inputs[3], 5, inputs[3].shape[-1], kwargs["weight_decay"]);

                with tf.variable_scope('ang', values=kwargs["values"]):
                    sela, ffa = neko_abc_collection.cat_coabcn(inputs[3],inputs[3], 3, inputs[3].shape[-1],kwargs["weight_decay"]);

                F_score = slim.conv2d(ffc, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                # 4 channel of axis aligned bbox and 1 channel rotation angle
                geo_map = slim.conv2d(ffr, 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * kwargs["text_scale"];
                angle_map = (slim.conv2d(ffa, 1, 1, activation_fn=tf.nn.sigmoid,
                                         normalizer_fn=None) - 0.5) * np.pi / 2  # angle is between [-45, 45]
                F_geometry = tf.concat([geo_map, angle_map], axis=-1)

        return  F_score,F_geometry,tf.concat([selc,selr,sela],axis=-1);

class cat_abcah_slime_east_parser:
    #normalized as attention is all you need
    @staticmethod
    def cat_abca(feature, cnt_b, BS,weight_decay):
        #KQ^T
        aks=tf.get_variable("anchor_keys",shape=[1,1,feature.shape[-1],cnt_b],
                               initializer=tf.initializers.glorot_normal,
                               regularizer=slim.l2_regularizer(weight_decay));
        naks=aks/(tf.norm(aks,axis=2,ord=2,keepdims=1)+0.0009);

        selector = tf.nn.conv2d(feature,naks,strides=(1,1,1,1),padding="VALID");
        selector = tf.nn.softmax(selector, axis=-1);
        selector = tf.expand_dims(selector, -2);

        to_select = slim.conv2d(feature, cnt_b * BS, 1);
        shp = tf.shape(to_select);
        to_select = tf.expand_dims(to_select, -2);
        to_select = tf.reshape(to_select, [shp[0], shp[1], shp[2], BS, cnt_b]);

        print(selector.shape);
        selected = to_select * selector;
        return selector, tf.reduce_sum(selected, axis=-1);
    def model(this,inputs,is_training,**kwargs):
        # result parsing should not be bounded to feature fusion
        with tf.variable_scope('result_parser_ABC', values=kwargs["values"]):
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
                with tf.variable_scope('cls', values=kwargs["values"]):
                    selc, ffc = this.cat_abca(inputs[3], 9, inputs[3].shape[-1],kwargs["weight_decay"]);

                with tf.variable_scope('reg', values=kwargs["values"]):
                    selr, ffr = this.cat_abca(inputs[3], 9, inputs[3].shape[-1], kwargs["weight_decay"]);

                with tf.variable_scope('ang', values=kwargs["values"]):
                    sela, ffa = this.cat_abca(inputs[3], 9, inputs[3].shape[-1],kwargs["weight_decay"]);

                F_score = slim.conv2d(ffc, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                # 4 channel of axis aligned bbox and 1 channel rotation angle
                geo_map = slim.conv2d(ffr, 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * kwargs["text_scale"];
                angle_map = (slim.conv2d(ffa, 1, 1, activation_fn=tf.nn.sigmoid,
                                         normalizer_fn=None) - 0.5) * np.pi / 2  # angle is between [-45, 45]
                F_geometry = tf.concat([geo_map, angle_map], axis=-1)

        return  F_score,F_geometry,tf.concat([selc,selr,sela],axis=-1);

class cat_abcf_slime_east_parser:
    @staticmethod
    def cat_abcf(feature, cnt_b, BS):
        selector = slim.conv2d(feature, cnt_b, 1, activation_fn=None)/math.sqrt(cnt_b);

        selector = tf.nn.softmax(selector, axis=-1);
        selector = tf.expand_dims(selector, -2);

        to_select = slim.conv2d(feature, cnt_b * BS, 1);
        shp = tf.shape(to_select);
        to_select = tf.expand_dims(to_select, -2);
        to_select = tf.reshape(to_select, [shp[0], shp[1], shp[2], BS, cnt_b]);

        print(selector.shape);
        selected = to_select * selector;
        return selector, tf.reduce_sum(selected, axis=-1);
    def model(this,inputs,is_training,**kwargs):
        # result parsing should not be bounded to feature fusion
        with tf.variable_scope('result_parser_ABC', values=kwargs["values"]):
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
                selc, ffc = this.cat_abcf(inputs[3], 3, inputs[3].shape[-1]);

                selr, ffr = this.cat_abcf(inputs[3], 5, inputs[3].shape[-1]);

                sela, ffa = this.cat_abcf(inputs[3], 3, inputs[3].shape[-1]);

                F_score = slim.conv2d(ffc, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                # 4 channel of axis aligned bbox and 1 channel rotation angle
                geo_map = slim.conv2d(ffr, 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * kwargs["text_scale"];
                angle_map = (slim.conv2d(ffa, 1, 1, activation_fn=tf.nn.sigmoid,
                                         normalizer_fn=None) - 0.5) * np.pi / 2  # angle is between [-45, 45]
                F_geometry = tf.concat([geo_map, angle_map], axis=-1)

        return  F_score,F_geometry,tf.concat([selc,selr,sela],axis=-1);

class cat_abcf2_slime_east_parser:
    @staticmethod
    def cat_abcf(feature, cnt_b, BS):
        selector = slim.conv2d(feature, cnt_b, 1, activation_fn=None);

        selector = tf.nn.softmax(selector, axis=-1);
        selector = tf.expand_dims(selector, -2);

        to_select = slim.conv2d(feature, cnt_b * BS, 1);
        shp = tf.shape(to_select);
        to_select = tf.expand_dims(to_select, -2);
        to_select = tf.reshape(to_select, [shp[0], shp[1], shp[2], BS, cnt_b]);

        print(selector.shape);
        selected = to_select * selector;
        return selector, tf.reduce_sum(selected, axis=-1);
    def model(this,inputs,is_training,**kwargs):
        # result parsing should not be bounded to feature fusion
        with tf.variable_scope('result_parser_ABC', values=kwargs["values"]):
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
                selc, ffc = this.cat_abcf(inputs[3], 3, inputs[3].shape[-1]);

                selr, ffr = this.cat_abcf(inputs[3], 5, inputs[3].shape[-1]);

                sela, ffa = this.cat_abcf(inputs[3], 3, inputs[3].shape[-1]);

                F_score = slim.conv2d(ffc, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                # 4 channel of axis aligned bbox and 1 channel rotation angle
                geo_map = slim.conv2d(ffr, 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * kwargs["text_scale"];
                angle_map = (slim.conv2d(ffa, 1, 1, activation_fn=tf.nn.sigmoid,
                                         normalizer_fn=None) - 0.5) * np.pi / 2  # angle is between [-45, 45]
                F_geometry = tf.concat([geo_map, angle_map], axis=-1)

        return  F_score,F_geometry,tf.concat([selc,selr,sela],axis=-1);

class cat_abchb_nss_slime_east_parser:
    @staticmethod
    def cat_abchb(feature, cnt_b, BS):
        selector = slim.conv2d(feature, cnt_b, 1, activation_fn=None, normalizer_fn=None)
        selector = tf.nn.softmax(selector, axis=-1);
        selector = tf.expand_dims(selector, -2);

        to_select = slim.conv2d(feature, cnt_b * BS, 1);

        sb_i = tf.reduce_mean(to_select, [1, 2]);
        sb = slim.fully_connected(sb_i, 1, tf.nn.sigmoid);
        tf.reshape(sb, (-1, 1, 1, 1));
        selector += sb;

        shp = tf.shape(to_select);
        to_select = tf.expand_dims(to_select, -2);
        to_select = tf.reshape(to_select, [shp[0], shp[1], shp[2], BS, cnt_b]);

        print(selector.shape);
        selected = to_select * selector;
        return selector, tf.reduce_sum(selected, axis=-1);
    def model(this,inputs,is_training,**kwargs):
        # result parsing should not be bounded to feature fusion
        with tf.variable_scope('result_parser_ABC', values=kwargs["values"]):
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
                selc, ffc = this.cat_abchb(inputs[3], 3, inputs[3].shape[-1]);

                selr, ffr = this.cat_abchb(inputs[3], 5, inputs[3].shape[-1]);

                sela, ffa = this.cat_abchb(inputs[3], 3, inputs[3].shape[-1]);

                F_score = slim.conv2d(ffc, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                # 4 channel of axis aligned bbox and 1 channel rotation angle
                geo_map = slim.conv2d(ffr, 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * kwargs["text_scale"];
                angle_map = (slim.conv2d(ffa, 1, 1, activation_fn=tf.nn.sigmoid,
                                         normalizer_fn=None) - 0.5) * np.pi / 2  # angle is between [-45, 45]
                F_geometry = tf.concat([geo_map, angle_map], axis=-1)

        return  F_score,F_geometry,tf.concat([selc,selr,sela],axis=-1);

class cat_abc_dist_slime_east_parser:
    @staticmethod
    def cat_abc(feature, cnt_b, BS):
        selector = slim.conv2d(feature, cnt_b, 1, activation_fn=None, normalizer_fn=None)
        selector = tf.nn.softmax(selector, axis=-1);
        selector = tf.expand_dims(selector, -2);

        to_select = slim.conv2d(feature, cnt_b * BS, 1);
        shp = tf.shape(to_select);
        to_select = tf.expand_dims(to_select, -2);
        to_select = tf.reshape(to_select, [shp[0], shp[1], shp[2], BS, cnt_b]);

        print(selector.shape);
        selected = to_select * selector;
        b0 = to_select[:, :, :, :, 0];
        b1 = to_select[:, :, :, :, 1];
        b01 = tf.reduce_sum((b0 * b1), axis=-1);
        b00 = tf.norm(b0, axis=-1);
        b11 = tf.norm(b1, axis=-1);
        return  tf.expand_dims(tf.expand_dims(b01 / b00 / b11, -1), -1), tf.reduce_sum(selected, axis=-1);
    def model(this,inputs,is_training,**kwargs):
        # result parsing should not be bounded to feature fusion
        with tf.variable_scope('result_parser_ABC', values=kwargs["values"]):
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
                sel, ff = this.cat_abc(inputs[3], 9, inputs[3].shape[-1]);

                F_score = slim.conv2d(ff, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                # 4 channel of axis aligned bbox and 1 channel rotation angle
                geo_map = slim.conv2d(ff, 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * kwargs["text_scale"];
                angle_map = (slim.conv2d(ff, 1, 1, activation_fn=tf.nn.sigmoid,
                                         normalizer_fn=None) - 0.5) * np.pi / 2  # angle is between [-45, 45]
                F_geometry = tf.concat([geo_map, angle_map], axis=-1)

        return  F_score,F_geometry,sel;


class cat_abc_slime_east_parserfwonegc:
    @staticmethod
    def cat_abc(feature, cnt_b, BS,id):
        selector = slim.conv2d(feature, cnt_b, 1, activation_fn=None, normalizer_fn=None)
        selector = tf.nn.softmax(selector, axis=-1);
        selector = tf.expand_dims(selector, -2);

        to_select = slim.conv2d(feature, cnt_b * BS, 1);
        shp = tf.shape(to_select);
        to_select = tf.expand_dims(to_select, -2);
        to_select = tf.reshape(to_select, [shp[0], shp[1], shp[2], BS, cnt_b]);

        print(selector.shape);
        selected = to_select * selector;

        selected_f = selected[:,:,:,:,id];
        return selector, tf.reduce_sum(selected, axis=-1),selected_f;
    def model(this,inputs,is_training,**kwargs):
        # result parsing should not be bounded to feature fusion
        with tf.variable_scope('result_parser_ABC', values=kwargs["values"]):
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
                sel, ff,fff = this.cat_abc(inputs[3], 9, inputs[3].shape[-1],kwargs["id"]);

                F_score = slim.conv2d(ff, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                # 4 channel of axis aligned bbox and 1 channel rotation angle
                geo_map = slim.conv2d(fff, 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * kwargs["text_scale"];
                angle_map = (slim.conv2d(fff, 1, 1, activation_fn=tf.nn.sigmoid,
                                         normalizer_fn=None) - 0.5) * np.pi / 2  # angle is between [-45, 45]
                F_geometry = tf.concat([geo_map, angle_map], axis=-1)

        return  F_score,F_geometry,sel;


class cat_abc_slime_east_parserfwone:
    @staticmethod
    def cat_abc(feature, cnt_b, BS,id):
        selector = slim.conv2d(feature, cnt_b, 1, activation_fn=None, normalizer_fn=None)
        selector = tf.nn.softmax(selector, axis=-1);
        selector = tf.expand_dims(selector, -2);

        to_select = slim.conv2d(feature, cnt_b * BS, 1);
        shp = tf.shape(to_select);
        to_select = tf.expand_dims(to_select, -2);
        to_select = tf.reshape(to_select, [shp[0], shp[1], shp[2], BS, cnt_b]);

        print(selector.shape);
        selected = to_select[:,:,:,:,id];

        return selector, selected;
    def model(this,inputs,is_training,**kwargs):
        # result parsing should not be bounded to feature fusion
        with tf.variable_scope('result_parser_ABC', values=kwargs["values"]):
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
                sel, ff = this.cat_abc(inputs[3], 9, inputs[3].shape[-1],kwargs["id"]);

                F_score = slim.conv2d(ff, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                # 4 channel of axis aligned bbox and 1 channel rotation angle
                geo_map = slim.conv2d(ff, 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * kwargs["text_scale"];
                angle_map = (slim.conv2d(ff, 1, 1, activation_fn=tf.nn.sigmoid,
                                         normalizer_fn=None) - 0.5) * np.pi / 2  # angle is between [-45, 45]
                F_geometry = tf.concat([geo_map, angle_map], axis=-1)

        return  F_score,F_geometry,sel;


class cat_abc_slime_east_parserfw025:
    @staticmethod
    def cat_abc(feature, cnt_b, BS):
        selector = slim.conv2d(feature, cnt_b, 1, activation_fn=None, normalizer_fn=None)
        selector = tf.nn.softmax(selector, axis=-1);
        selector = tf.expand_dims(selector, -2);

        to_select = slim.conv2d(feature, cnt_b * BS, 1);
        shp = tf.shape(to_select);
        to_select = tf.expand_dims(to_select, -2);
        to_select = tf.reshape(to_select, [shp[0], shp[1], shp[2], BS, cnt_b]);

        print(selector.shape);
        selected = to_select *(1./cnt_b);
        return selector, tf.reduce_sum(selected, axis=-1);
    def model(this,inputs,is_training,**kwargs):
        # result parsing should not be bounded to feature fusion
        with tf.variable_scope('result_parser_ABC', values=kwargs["values"]):
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
                sel, ff = this.cat_abc(inputs[3], 9, inputs[3].shape[-1]);

                F_score = slim.conv2d(ff, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                # 4 channel of axis aligned bbox and 1 channel rotation angle
                geo_map = slim.conv2d(ff, 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * kwargs["text_scale"];
                angle_map = (slim.conv2d(ff, 1, 1, activation_fn=tf.nn.sigmoid,
                                         normalizer_fn=None) - 0.5) * np.pi / 2  # angle is between [-45, 45]
                F_geometry = tf.concat([geo_map, angle_map], axis=-1)

        return  F_score,F_geometry,sel;


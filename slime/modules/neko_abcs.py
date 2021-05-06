import tensorflow as tf;
from tensorflow.contrib import slim
import math;
try:
    import tensorflow.initializers.glorot_normal as xavier
except:
    # Old Tensorflows has no tensorflow.initializers.glorot_normal...
    xavier=tf.glorot_normal_initializer()

class neko_abc_collection:
    @staticmethod
    def cat_mlp_af(feature,BS,hidden):
        to_select = slim.conv2d(feature, hidden, 1);
        to_select = slim.conv2d(to_select, BS, 1);
        return to_select;


    @staticmethod
    def cat_coabcs(feature, sfeature, cnt_b, BS, weight_decay,hidden):
        # QK^T-> bn(QK^T)/Sqrt(d_k)

        aks = tf.get_variable("anchor_keys", shape=[1, 1, sfeature.shape[-1], cnt_b],
                              initializer=xavier,
                              regularizer=slim.l2_regularizer(weight_decay));
        naks = aks / (tf.norm(aks, axis=2, ord=2, keepdims=1) + 0.0009);

        selector = tf.nn.conv2d(sfeature, naks, strides=(1, 1, 1, 1), padding="SAME");

        selector = tf.nn.softmax(selector, axis=-1);
        selector = tf.expand_dims(selector, -2);
        c=[];
        for i in range(cnt_b):
            c.append(neko_abc_collection.cat_mlp_af(feature,BS,hidden));
        to_select=tf.stack(c,-1);
        print(selector.shape);
        selected = to_select * selector;
        return selector, tf.reduce_sum(selected, axis=-1);
    @staticmethod
    def cat_coabca(feature, sfeature, cnt_b, BS, weight_decay):
        # QK^T-> bn(QK^T)/Sqrt(d_k)

        aks = tf.get_variable("anchor_keys", shape=[1, 1, sfeature.shape[-1], cnt_b],
                              initializer=xavier,
                              regularizer=slim.l2_regularizer(weight_decay));
        naks = aks / (tf.norm(aks, axis=2, ord=2, keepdims=1) + 0.0009);

        selector = tf.nn.conv2d(sfeature, naks, strides=(1, 1, 1, 1), padding="SAME");

        selector = tf.nn.softmax(selector, axis=-1);
        selector = tf.expand_dims(selector, -2);

        to_select = slim.conv2d(feature, cnt_b * BS, 1);

        shp = tf.shape(to_select);
        to_select = tf.expand_dims(to_select, -2);
        to_select = tf.reshape(to_select, [shp[0], shp[1], shp[2], BS, cnt_b]);

        print(selector.shape);
        selected = to_select * selector;
        return selector, tf.reduce_sum(selected, axis=-1);

    @staticmethod
    def cat_coabca_nd(feature, sfeature, cnt_b, BS, weight_decay):
        # QK^T-> bn(QK^T)/Sqrt(d_k)

        aks = tf.get_variable("anchor_keys", shape=[1, 1, sfeature.shape[-1], cnt_b],
                              initializer=xavier,
                              regularizer=slim.l2_regularizer(weight_decay));
        naks = aks / (tf.norm(aks, axis=2, ord=2, keepdims=1) + 0.0009);

        selector = tf.nn.conv2d(sfeature, naks, strides=(1, 1, 1, 1), padding="SAME");

        selector = tf.nn.softmax(selector, axis=-1);
        selector = tf.expand_dims(selector, -2);

        to_select = slim.conv2d(feature, cnt_b * BS, 1);

        shp = tf.shape(to_select);
        to_select = tf.expand_dims(to_select, -2);
        to_select = tf.reshape(to_select, [shp[0], shp[1], shp[2], BS, cnt_b]);

        print(selector.shape);
        selected = to_select * selector;
        return selector, tf.reduce_sum(selected, axis=-1);

    @staticmethod
    def cat_coabcn(feature, sfeature, cnt_b, BS, weight_decay):
        # QK^T-> bn(QK^T)/Sqrt(d_k)


        aks = tf.get_variable("anchor_keys", shape=[1, 1, sfeature.shape[-1], cnt_b],
                              initializer=xavier,
                              regularizer=slim.l2_regularizer(weight_decay));
        naks = aks / (tf.norm(aks, axis=2, ord=2, keepdims=1) + 0.0009);

        nsfeature=sfeature/(tf.norm(sfeature, axis=-1, ord=2, keepdims=1) + 0.0009);

        selector = tf.nn.conv2d(nsfeature, naks, strides=(1, 1, 1, 1), padding="SAME")*3; # make it sharper

        selector = tf.nn.softmax(selector, axis=-1);
        selector = tf.expand_dims(selector, -2);

        to_select = slim.conv2d(feature, cnt_b * BS, 1);

        shp = tf.shape(to_select);
        to_select = tf.expand_dims(to_select, -2);
        to_select = tf.reshape(to_select, [shp[0], shp[1], shp[2], BS, cnt_b]);

        print(selector.shape);
        selected = to_select * selector;
        return selector, tf.reduce_sum(selected, axis=-1);
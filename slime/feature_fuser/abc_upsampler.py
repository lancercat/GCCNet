import tensorflow as tf;
from tensorflow.contrib import slim;
from neko_dogoo_v3.utils_module import dogoo_ops;
from slime.feature_fuser.abstract_unpooler import abstract_unpool;
from slime.modules.neko_abcs import neko_abc_collection;
import math;
class cat_abc_unpool(abstract_unpool):
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

    def model(this,inputs,is_training,num_outputs = [128, 128, 64, 32],**kwargs):
        num_basis = [8, 4, 4, 4];
        with tf.variable_scope('feature_fusion', values=kwargs["values"]):
            with this.get_slime_arg_scope(is_training,weight_decay=kwargs["weight_decay"]):
                g = [];
                h = [];
                l = [];
                w=[];
                for i in range(len(num_outputs)):
                    print('Shape of f_{} {}'.format(i, inputs[i].shape))
                    g.append(None);
                    h.append(None);
                    l.append(None);
                    w.append(None);
                for i in range(len(num_outputs)):
                    w[i],l[i]=this.cat_abc(inputs[i],num_basis[i],num_outputs[i])
                    if i == 0:
                        h[i] = l[i];
                    else:
                        c1_1 = slim.conv2d(tf.concat([g[i - 1], l[i]], axis=-1), num_outputs[i], 1)
                        h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                    if i <= len(num_outputs)-2:
                        g[i] = dogoo_ops.unpool(h[i])
                    else:
                        g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                    print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))
        return g;

class cat_coabc_unpool(abstract_unpool):
    @staticmethod
    def cat_coabc(feature,sfeature, cnt_b, BS):
        selector = slim.conv2d(sfeature, cnt_b, 1, activation_fn=None, normalizer_fn=None)
        selector = tf.nn.softmax(selector, axis=-1);
        selector = tf.expand_dims(selector, -2);

        to_select = slim.conv2d(feature, cnt_b * BS, 1);
        shp = tf.shape(to_select);
        to_select = tf.expand_dims(to_select, -2);
        to_select = tf.reshape(to_select, [shp[0], shp[1], shp[2], BS, cnt_b]);

        print(selector.shape);
        selected = to_select * selector;
        return selector, tf.reduce_sum(selected, axis=-1);

    def model(this,inputs,is_training,num_outputs = [128, 128, 64, 32],**kwargs):
        num_basis = [8, 4, 4, 4];
        with tf.variable_scope('feature_fusion', values=kwargs["values"]):
            with this.get_slime_arg_scope(is_training,weight_decay=kwargs["weight_decay"]):
                g = [];
                h = [];
                l = [];
                w=[];
                for i in range(len(num_outputs)):
                    print('Shape of f_{} {}'.format(i, inputs[i].shape))
                    g.append(None);
                    h.append(None);
                    l.append(None);
                    w.append(None);
                for i in range(len(num_outputs)):
                    if i == 0:
                        w[i], l[i] = this.cat_coabc(inputs[i],inputs[i], num_basis[i], num_outputs[i])
                        h[i] = l[i];
                    else:
                        w[i], l[i] = this.cat_coabc(inputs[i],g[i-1], num_basis[i], num_outputs[i])
                        c1_1 = slim.conv2d(tf.concat([g[i - 1], l[i]], axis=-1), num_outputs[i], 1)
                        h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                    if i <= len(num_outputs)-2:
                        g[i] = dogoo_ops.unpool(h[i])
                    else:
                        g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                    print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))
        return g;
class cat_coabch_unpool(abstract_unpool):
    @staticmethod
    def cat_coabc(feature,sfeature, cnt_b, BS):
        selector = slim.conv2d(sfeature, cnt_b, 1, activation_fn=None, normalizer_fn=None)
        selector = tf.nn.softmax(selector, axis=-1);
        selector = tf.expand_dims(selector, -2);

        to_select = slim.conv2d(feature, cnt_b * BS, 1);

        shp = tf.shape(to_select);
        to_select = tf.expand_dims(to_select, -2);
        to_select = tf.reshape(to_select, [shp[0], shp[1], shp[2], BS, cnt_b]);

        print(selector.shape);
        selected = to_select * selector;
        return selector, tf.reduce_sum(selected, axis=-1);

    def model(this,inputs,is_training,num_outputs = [256, 128, 64, 32],**kwargs):
        num_basis = [8, 4, 4, 4];
        with tf.variable_scope('feature_fusion', values=kwargs["values"]):
            with this.get_slime_arg_scope(is_training,weight_decay=kwargs["weight_decay"]):
                g = [];
                h = [];
                l = [];
                w=[];
                for i in range(len(num_outputs)):
                    print('Shape of f_{} {}'.format(i, inputs[i].shape))
                    g.append(None);
                    h.append(None);
                    l.append(None);
                    w.append(None);
                for i in range(len(num_outputs)):
                    if i == 0:
                        w[i], l[i] = this.cat_coabc(inputs[i],inputs[i], num_basis[i], num_outputs[i])
                        h[i] = l[i];
                    else:
                        w[i], l[i] = this.cat_coabc(inputs[i],g[i-1], num_basis[i], num_outputs[i-1])
                        c1_1 = slim.conv2d(tf.concat([g[i - 1], l[i]], axis=-1), num_outputs[i-1], 1)
                        h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                    if i <= len(num_outputs)-2:
                        g[i] = dogoo_ops.unpool(h[i])
                    else:
                        g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                    print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))
        return g;


class cat_abcs_unpool(abstract_unpool):

    def model(this,inputs,is_training,num_outputs = [256, 128, 64, 32],**kwargs):
        num_basis = [8, 4, 4, 4];
        with tf.variable_scope('feature_fusion', values=kwargs["values"]):
            with this.get_slime_arg_scope(is_training,weight_decay=kwargs["weight_decay"]):
                g = [];
                h = [];
                l = [];
                w=[];
                for i in range(len(num_outputs)):
                    print('Shape of f_{} {}'.format(i, inputs[i].shape))
                    g.append(None);
                    h.append(None);
                    l.append(None);
                    w.append(None);
                for i in range(len(num_outputs)):
                    if i == 0:
                        with tf.variable_scope(str(i), values=kwargs["values"]):
                            w[i], l[i] = neko_abc_collection.cat_coabcs(inputs[i],inputs[i], num_basis[i], num_outputs[i],kwargs["weight_decay"], num_outputs[i]);
                        h[i] = l[i];
                    else:
                        with tf.variable_scope(str(i), values=kwargs["values"]):
                            w[i], l[i] = neko_abc_collection.cat_coabcs(inputs[i],g[i-1], num_basis[i], num_outputs[i-1],kwargs["weight_decay"], num_outputs[i]);

                        c1_1 = slim.conv2d(tf.concat([g[i - 1], l[i]], axis=-1), num_outputs[i-1], 1)
                        h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                    if i <= len(num_outputs)-2:
                        g[i] = dogoo_ops.unpool(h[i])
                    else:
                        g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                    print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))
        return g;

class cat_abca_unpool(abstract_unpool):


    def model(this,inputs,is_training,num_outputs = [256, 128, 64, 32],**kwargs):
        num_basis = [8, 4, 4, 4];
        with tf.variable_scope('feature_fusion', values=kwargs["values"]):
            with this.get_slime_arg_scope(is_training,weight_decay=kwargs["weight_decay"]):
                g = [];
                h = [];
                l = [];
                w=[];
                for i in range(len(num_outputs)):
                    print('Shape of f_{} {}'.format(i, inputs[i].shape))
                    g.append(None);
                    h.append(None);
                    l.append(None);
                    w.append(None);
                for i in range(len(num_outputs)):
                    if i == 0:
                        with tf.variable_scope(str(i), values=kwargs["values"]):
                            w[i], l[i] = neko_abc_collection.cat_coabca(inputs[i],inputs[i], num_basis[i], num_outputs[i],weight_decay=kwargs["weight_decay"]);
                        h[i] = l[i];
                    else:
                        with tf.variable_scope(str(i), values=kwargs["values"]):
                            w[i], l[i] = neko_abc_collection.cat_coabca(inputs[i],g[i-1], num_basis[i], num_outputs[i-1],weight_decay=kwargs["weight_decay"]);

                        c1_1 = slim.conv2d(tf.concat([g[i - 1], l[i]], axis=-1), num_outputs[i-1], 1)
                        h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                    if i <= len(num_outputs)-2:
                        g[i] = dogoo_ops.unpool(h[i])
                    else:
                        g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                    print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))
        return g;
class cat_abcn_unpool(abstract_unpool):


    def model(this,inputs,is_training,num_outputs = [256, 128, 64, 32],**kwargs):
        num_basis = [8, 4, 4, 4];
        with tf.variable_scope('feature_fusion', values=kwargs["values"]):
            with this.get_slime_arg_scope(is_training,weight_decay=kwargs["weight_decay"]):
                g = [];
                h = [];
                l = [];
                w=[];
                for i in range(len(num_outputs)):
                    print('Shape of f_{} {}'.format(i, inputs[i].shape))
                    g.append(None);
                    h.append(None);
                    l.append(None);
                    w.append(None);
                for i in range(len(num_outputs)):
                    if i == 0:
                        with tf.variable_scope(str(i), values=kwargs["values"]):
                            w[i], l[i] = neko_abc_collection.cat_coabcn(inputs[i],inputs[i], num_basis[i], num_outputs[i],weight_decay=kwargs["weight_decay"]);
                        h[i] = l[i];
                    else:
                        with tf.variable_scope(str(i), values=kwargs["values"]):
                            w[i], l[i] = neko_abc_collection.cat_coabcn(inputs[i],g[i-1], num_basis[i], num_outputs[i-1],weight_decay=kwargs["weight_decay"]);

                        c1_1 = slim.conv2d(tf.concat([g[i - 1], l[i]], axis=-1), num_outputs[i-1], 1)
                        h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                    if i <= len(num_outputs)-2:
                        g[i] = dogoo_ops.unpool(h[i])
                    else:
                        g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                    print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))
        return g;

class cat_abcf_unpool(abstract_unpool):
    @staticmethod
    def cat_coabcf(feature,sfeature, cnt_b, BS):
        #QK^T-> bn(QK^T)/Sqrt(d_k)
        selector = slim.conv2d(sfeature, cnt_b, 1, activation_fn=None)/math.sqrt(cnt_b);

        selector = tf.nn.softmax(selector, axis=-1);
        selector = tf.expand_dims(selector, -2);

        to_select = slim.conv2d(feature, cnt_b * BS, 1);

        shp = tf.shape(to_select);
        to_select = tf.expand_dims(to_select, -2);
        to_select = tf.reshape(to_select, [shp[0], shp[1], shp[2], BS, cnt_b]);

        print(selector.shape);
        selected = to_select * selector;
        return selector, tf.reduce_sum(selected, axis=-1);

    def model(this,inputs,is_training,num_outputs = [256, 128, 64, 32],**kwargs):
        num_basis = [8, 4, 4, 4];
        with tf.variable_scope('feature_fusion', values=kwargs["values"]):
            with this.get_slime_arg_scope(is_training,weight_decay=kwargs["weight_decay"]):
                g = [];
                h = [];
                l = [];
                w=[];
                for i in range(len(num_outputs)):
                    print('Shape of f_{} {}'.format(i, inputs[i].shape))
                    g.append(None);
                    h.append(None);
                    l.append(None);
                    w.append(None);
                for i in range(len(num_outputs)):
                    if i == 0:
                        w[i], l[i] = this.cat_coabcf(inputs[i],inputs[i], num_basis[i], num_outputs[i])
                        h[i] = l[i];
                    else:
                        w[i], l[i] = this.cat_coabcf(inputs[i],g[i-1], num_basis[i], num_outputs[i-1])
                        c1_1 = slim.conv2d(tf.concat([g[i - 1], l[i]], axis=-1), num_outputs[i-1], 1)
                        h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                    if i <= len(num_outputs)-2:
                        g[i] = dogoo_ops.unpool(h[i])
                    else:
                        g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                    print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))
        return g;

class cat_abcf2_unpool(abstract_unpool):
    @staticmethod
    def cat_coabcf2(feature,sfeature, cnt_b, BS):
        #QK^T-> bn(QK^T)
        selector = slim.conv2d(sfeature, cnt_b, 1, activation_fn=None);

        selector = tf.nn.softmax(selector, axis=-1);
        selector = tf.expand_dims(selector, -2);

        to_select = slim.conv2d(feature, cnt_b * BS, 1);

        shp = tf.shape(to_select);
        to_select = tf.expand_dims(to_select, -2);
        to_select = tf.reshape(to_select, [shp[0], shp[1], shp[2], BS, cnt_b]);

        print(selector.shape);
        selected = to_select * selector;
        return selector, tf.reduce_sum(selected, axis=-1);

    def model(this,inputs,is_training,num_outputs = [256, 128, 64, 32],**kwargs):
        num_basis = [8, 4, 4, 4];
        with tf.variable_scope('feature_fusion', values=kwargs["values"]):
            with this.get_slime_arg_scope(is_training,weight_decay=kwargs["weight_decay"]):
                g = [];
                h = [];
                l = [];
                w=[];
                for i in range(len(num_outputs)):
                    print('Shape of f_{} {}'.format(i, inputs[i].shape))
                    g.append(None);
                    h.append(None);
                    l.append(None);
                    w.append(None);
                for i in range(len(num_outputs)):
                    if i == 0:
                        w[i], l[i] = this.cat_coabcf2(inputs[i],inputs[i], num_basis[i], num_outputs[i])
                        h[i] = l[i];
                    else:
                        w[i], l[i] = this.cat_coabcf2(inputs[i],g[i-1], num_basis[i], num_outputs[i-1])
                        c1_1 = slim.conv2d(tf.concat([g[i - 1], l[i]], axis=-1), num_outputs[i-1], 1)
                        h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                    if i <= len(num_outputs)-2:
                        g[i] = dogoo_ops.unpool(h[i])
                    else:
                        g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                    print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))
        return g;

class cat_coabchb_unpool(abstract_unpool):
    @staticmethod
    def cat_coabc(feature,sfeature, cnt_b, BS):
        selector = slim.conv2d(sfeature, cnt_b, 1, activation_fn=None, normalizer_fn=None)
        selector = tf.nn.softmax(selector, axis=-1);
        selector = tf.expand_dims(selector, -2);

        to_select = slim.conv2d(feature, cnt_b * BS, 1);

        sb_i=tf.reduce_mean(to_select,[1,2]);
        sb=slim.fully_connected(sb_i,1,tf.nn.sigmoid);
        tf.reshape(sb,(-1,1,1,1));
        selector+=sb;

        shp = tf.shape(to_select);
        to_select = tf.expand_dims(to_select, -2);
        to_select = tf.reshape(to_select, [shp[0], shp[1], shp[2], BS, cnt_b]);

        print(selector.shape);
        selected = to_select * selector;
        return selector, tf.reduce_sum(selected, axis=-1);


    def model(this,inputs,is_training,num_outputs = [256, 128, 64, 32],**kwargs):
        num_basis = [8, 4, 4, 4];
        with tf.variable_scope('feature_fusion', values=kwargs["values"]):
            with this.get_slime_arg_scope(is_training,weight_decay=kwargs["weight_decay"]):
                g = [];
                h = [];
                l = [];
                w=[];
                for i in range(len(num_outputs)):
                    print('Shape of f_{} {}'.format(i, inputs[i].shape))
                    g.append(None);
                    h.append(None);
                    l.append(None);
                    w.append(None);
                for i in range(len(num_outputs)):
                    if i == 0:
                        w[i], l[i] = this.cat_coabc(inputs[i],inputs[i], num_basis[i], num_outputs[i])
                        h[i] = l[i];
                    else:
                        w[i], l[i] = this.cat_coabc(inputs[i],g[i-1], num_basis[i], num_outputs[i-1])
                        c1_1 = slim.conv2d(tf.concat([g[i - 1], l[i]], axis=-1), num_outputs[i-1], 1)
                        h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                    if i <= len(num_outputs)-2:
                        g[i] = dogoo_ops.unpool(h[i])
                    else:
                        g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                    print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))
        return g;

class cat_abc_unpool_d(abstract_unpool):
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

    def model(this,inputs,is_training,num_outputs = [128, 128, 64, 32],**kwargs):
        num_basis = [8, 4, 4, 4];
        with tf.variable_scope('feature_fusion', values=kwargs["values"]):
            with this.get_slime_arg_scope(is_training,weight_decay=kwargs["weight_decay"]):
                g = [];
                h = [];
                l = [];
                w=[];
                for i in range(len(num_outputs)):
                    print('Shape of f_{} {}'.format(i, inputs[i].shape))
                    g.append(None);
                    h.append(None);
                    l.append(None);
                    w.append(None);
                for i in range(len(num_outputs)):
                    w[i],l[i]=this.cat_abc(inputs[i],num_basis[i],num_outputs[i])
                    if i == 0:
                        h[i] = l[i];
                    else:
                        c1_1 = slim.conv2d(tf.concat([g[i - 1], l[i]], axis=-1), num_outputs[i], 1)
                        h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                    if i <= len(num_outputs)-2:
                        g[i] = dogoo_ops.unpool(h[i])
                    else:
                        g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                    print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))
        return g,w;
class cat_coabc_unpool_d(abstract_unpool):
    @staticmethod
    def cat_coabc(feature,sfeature, cnt_b, BS):
        selector = slim.conv2d(sfeature, cnt_b, 1, activation_fn=None, normalizer_fn=None)
        selector = tf.nn.softmax(selector, axis=-1);
        selector = tf.expand_dims(selector, -2);

        to_select = slim.conv2d(feature, cnt_b * BS, 1);
        shp = tf.shape(to_select);
        to_select = tf.expand_dims(to_select, -2);
        to_select = tf.reshape(to_select, [shp[0], shp[1], shp[2], BS, cnt_b]);

        print(selector.shape);
        selected = to_select * selector;
        return selector, tf.reduce_sum(selected, axis=-1);

    def model(this,inputs,is_training,num_outputs = [128, 128, 64, 32],**kwargs):
        num_basis = [8, 4, 4, 4];
        with tf.variable_scope('feature_fusion', values=kwargs["values"]):
            with this.get_slime_arg_scope(is_training,weight_decay=kwargs["weight_decay"]):
                g = [];
                h = [];
                l = [];
                w=[];
                for i in range(len(num_outputs)):
                    print('Shape of f_{} {}'.format(i, inputs[i].shape))
                    g.append(None);
                    h.append(None);
                    l.append(None);
                    w.append(None);
                for i in range(len(num_outputs)):
                    if i == 0:
                        w[i], l[i] = this.cat_coabc(inputs[i],inputs[i], num_basis[i], num_outputs[i])
                        h[i] = l[i];
                    else:
                        w[i], l[i] = this.cat_coabc(inputs[i],g[i-1], num_basis[i], num_outputs[i])
                        c1_1 = slim.conv2d(tf.concat([g[i - 1], l[i]], axis=-1), num_outputs[i], 1)
                        h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                    if i <= len(num_outputs)-2:
                        g[i] = dogoo_ops.unpool(h[i])
                    else:
                        g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                    print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))
        return g,w;

class cat_coabc_unpool_dis(abstract_unpool):
    @staticmethod
    def cat_coabc(feature,sfeature, cnt_b, BS):
        selector = slim.conv2d(sfeature, cnt_b, 1, activation_fn=None, normalizer_fn=None)
        selector = tf.nn.softmax(selector, axis=-1);
        selector = tf.expand_dims(selector, -2);

        to_select = slim.conv2d(feature, cnt_b * BS, 1);
        shp = tf.shape(to_select);
        to_select = tf.expand_dims(to_select, -2);
        to_select = tf.reshape(to_select, [shp[0], shp[1], shp[2], BS, cnt_b]);

        b0=to_select[:,:,:,:,0];
        b1=to_select[:,:,:,:,1];
        b01=tf.reduce_sum((b0*b1),axis=-1);
        b00=tf.norm(b0,axis=-1);
        b11 = tf.norm(b1, axis=-1);

        print(selector.shape);
        selected = to_select * selector;
        return tf.expand_dims(tf.expand_dims(b01/b00/b11,-1),-1), tf.reduce_sum(selected, axis=-1);

    def model(this,inputs,is_training,num_outputs = [128, 128, 64, 32],**kwargs):
        num_basis = [8, 4, 4, 4];
        with tf.variable_scope('feature_fusion', values=kwargs["values"]):
            with this.get_slime_arg_scope(is_training,weight_decay=kwargs["weight_decay"]):
                g = [];
                h = [];
                l = [];
                w=[];
                for i in range(len(num_outputs)):
                    print('Shape of f_{} {}'.format(i, inputs[i].shape))
                    g.append(None);
                    h.append(None);
                    l.append(None);
                    w.append(None);
                for i in range(len(num_outputs)):
                    if i == 0:
                        w[i], l[i] = this.cat_coabc(inputs[i],inputs[i], num_basis[i], num_outputs[i])
                        h[i] = l[i];
                    else:
                        w[i], l[i] = this.cat_coabc(inputs[i],g[i-1], num_basis[i], num_outputs[i])
                        c1_1 = slim.conv2d(tf.concat([g[i - 1], l[i]], axis=-1), num_outputs[i], 1)
                        h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                    if i <= len(num_outputs)-2:
                        g[i] = dogoo_ops.unpool(h[i])
                    else:
                        g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                    print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))
        return g,w;


class cat_coabc_unpoolfw025(abstract_unpool):
    @staticmethod
    def cat_coabc(feature,sfeature, cnt_b, BS):
        selector = slim.conv2d(sfeature, cnt_b, 1, activation_fn=None, normalizer_fn=None)
        selector = tf.nn.softmax(selector, axis=-1);
        selector = tf.expand_dims(selector, -2);

        to_select = slim.conv2d(feature, cnt_b * BS, 1);
        shp = tf.shape(to_select);
        to_select = tf.expand_dims(to_select, -2);
        to_select = tf.reshape(to_select, [shp[0], shp[1], shp[2], BS, cnt_b]);

        print(selector.shape);
        selected = to_select* selector;
        return selector, tf.reduce_sum(selected, axis=-1);

    @staticmethod
    def cat_dcoabc(feature, sfeature, cnt_b, BS):
        selector = slim.conv2d(sfeature, cnt_b, 1, activation_fn=None, normalizer_fn=None)
        selector = tf.nn.softmax(selector, axis=-1);
        selector = tf.expand_dims(selector, -2);

        to_select = slim.conv2d(feature, cnt_b * BS, 1);
        shp = tf.shape(to_select);
        to_select = tf.expand_dims(to_select, -2);
        to_select = tf.reshape(to_select, [shp[0], shp[1], shp[2], BS, cnt_b]);

        print(selector.shape);
        selected = to_select * (1. / cnt_b);  # * selector;
        return selector, tf.reduce_sum(selected, axis=-1);

    def model(this,inputs,is_training,num_outputs = [128, 128, 64, 32],dabcids=[],**kwargs):
        num_basis = [8, 4, 4, 4];

        with tf.variable_scope('feature_fusion', values=kwargs["values"]):
            with this.get_slime_arg_scope(is_training,weight_decay=kwargs["weight_decay"]):
                g = [];
                h = [];
                l = [];
                w=[];
                for i in range(len(num_outputs)):
                    print('Shape of f_{} {}'.format(i, inputs[i].shape))
                    g.append(None);
                    h.append(None);
                    l.append(None);
                    w.append(None);
                for i in range(len(num_outputs)):
                    if i == 0:
                        if(4-i not in dabcids):
                            w[i], l[i] = this.cat_coabc(inputs[i],inputs[i], num_basis[i], num_outputs[i])
                        else:
                            print("disabling ",str(4-i));
                            w[i], l[i] = this.cat_dcoabc(inputs[i], inputs[i], num_basis[i], num_outputs[i])
                        h[i] = l[i];
                    else:
                        if(4-i not in dabcids):
                            w[i], l[i] = this.cat_coabc(inputs[i],g[i-1], num_basis[i], num_outputs[i])
                        else:
                            print("disabling ",str(4-i));
                            w[i], l[i] = this.cat_dcoabc(inputs[i],g[i-1], num_basis[i], num_outputs[i])
                        c1_1 = slim.conv2d(tf.concat([g[i - 1], l[i]], axis=-1), num_outputs[i], 1)
                        h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                    if i <= len(num_outputs)-2:
                        g[i] = dogoo_ops.unpool(h[i])
                    else:
                        g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                    print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))
        return g;

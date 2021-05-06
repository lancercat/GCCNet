import tensorflow as tf;
from tensorflow.contrib import slim;
from neko_dogoo_v3.utils_module import dogoo_ops;
from  slime.feature_fuser.abstract_unpooler import abstract_unpool;


class welf_slime_unpool(abstract_unpool):
    def model(this,inputs,is_training,num_outputs = [None, 128, 64, 32],**kwargs):
        with tf.variable_scope('feature_fusion', values=kwargs["values"]):
            with this.get_slime_arg_scope(is_training,weight_decay=kwargs["weight_decay"]):
                g = [];
                h = [];
                for i in range(len(num_outputs)):
                    print('Shape of f_{} {}'.format(i, inputs[i].shape))
                    g.append(None);
                    h.append(None);
                for i in range(len(num_outputs)):
                    if i == 0:
                        h[i] = inputs[i]
                    else:
                        c1_1 = slim.conv2d(tf.concat([g[i - 1], inputs[i]], axis=-1), num_outputs[i], 1)
                        h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                    if i <= len(num_outputs)-2:
                        g[i] = dogoo_ops.unpool(h[i])
                    else:
                        g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                    print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))
        return g;

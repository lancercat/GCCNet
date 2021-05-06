import tensorflow as tf;

from tensorflow.contrib import slim;
from utils.transplanter import get_list_of_variables_from_ckpt;
class abstract_model:
    PRFX = None;
    def restore_list(this,v,sess,ckpt_path):
        s = tf.train.Saver(var_list=v);
        s.restore(sess, ckpt_path);

    def restore(this,tr_prefix,sess,ckpt_path):
        v = slim.get_variables(tr_prefix,None,tf.GraphKeys.GLOBAL_VARIABLES);
        l = get_list_of_variables_from_ckpt(ckpt_path);
        print(l);
        s = tf.train.Saver(var_list=v);
        s.restore(sess,ckpt_path);
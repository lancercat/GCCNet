import tensorflow as tf;
from functools import partial;
import re;

from tensorflow.python.pywrap_tensorflow import NewCheckpointReader;
# Shall ya gimme file, I shall give ya names.
#  [A translated meme from some certain Chinese forums]
def get_list_of_variables_from_ckpt(ckpt_file):
    reader = NewCheckpointReader(ckpt_file);
    names = reader.get_variable_to_dtype_map();
    return list(names.keys());
def snr_remove(scope_name,name):
    return re.sub(scope_name,"",name);
def make_snr(scope_2rm):
    return partial(snr_remove,re.compile(r"^"+scope_2rm+"/"+r"|:[0-9]+$"));

def get_name_mapping(vars,names,snrfn):
    dict={};
    found=set();
    for variable in vars:
        vname=variable.name;
        sname=snrfn(vname);
        if (sname in names):
            dict[sname]=variable;
            found.add(vname);
        else:
            print(sname);
            print(vname);

    ssrc=set(names);
    unused =ssrc-found;
    print(unused);
    print(found);
    return dict;

def load_pretrained(vs, prefix, sess, path):
    in_ckpt = get_list_of_variables_from_ckpt(path);
    dpfn = make_snr(prefix);
    vlista = get_name_mapping(vs, in_ckpt, dpfn);
    saver = tf.train.Saver(var_list=vlista);
    saver.restore(sess, path);

def load_pretrained_torch(model, prefix, path):
    pass;

    # in_ckpt = get_list_of_variables_from_ckpt(path);
    # dpfn = make_snr(prefix);
    # vlista = get_name_mapping(vs, in_ckpt, dpfn);
    # saver = tf.train.Saver(var_list=vlista);
    # saver.restore(sess, path);
#
# if __name__ == '__main__':
#     names=get_list_of_variables_from_ckpt("../../../project_v_data/pretrained/resnet_v1_50.ckpt");
#     print(names);
#


import os;
import tensorflow as tf;
from utils.libpath import pathcfg;

from utils.notifier import nekoexpnotification;
from project_easts.loadouts.baseline import cat_mbaselineoom_evaluator
from project_easts.loadouts.cat_uniabc2a import cat_uniabc2aoom_evaluator,cat_uniabc2aoom_evaluator_res50;


def the_test(eval_fn,mpath,bbone,rpath,scale,key):
    eval=eval_fn(mpath,"0",False,backbone=bbone);
    res=eval.test_with_config(rpath,scale,key);
    return res;

def do_test(eval_fn,iter,mkey,key,scale=1.):
    mp = os.path.join(os.path.join(pathcfg.project_uniabc_data_root, mkey, eval_fn.FLAG), str(iter));
    res=the_test(eval_fn, mp, "densenet169",
                 os.path.join(pathcfg.project_uniabc_data_root,key,eval_fn.FLAG + str(iter)), scale, key);
    tf.reset_default_graph();
    nekoexpnotification().send_msg("lasercat_work@gmx.com","Done :"+eval_fn.FLAG +"f: " +str(res[0][2])+"Scale: "+str(scale)+"\n"+ str(iter)+"\n"+str(res));
    return res;



class test_ins:
    EVALS = [cat_uniabc2aoom_evaluator,cat_mbaselineoom_evaluator];

    START =400000;
    END = 400000 + 1;
    STEP = 10000;
    SCALS = [1];

    KEYS = [
        ("i15-train", "i15_eval"),

        # ("i15-train","i15_evalsnp"),
            # ("i15-train", "i15_evalm3"),
            # ("i15-train", "i15_evalm5"),
            # ("i15-train", "i15_evalm7"),
            # ("i15-train", "i15_evalsnp"),
        ];

    def doit(this):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0";
        for i in range(this.START, this.END, this.STEP):
            for scale in this.SCALS:
                for EVAL in this.EVALS:
                    EVAL.TH = 0.9;
                    for (mk,k) in this.KEYS:
                        do_test(EVAL, i,mk,k, scale);


if __name__ == '__main__':
    test_ins().doit();

import  tensorflow as tf;
from utils.cat_config import cat_config;
from project_easts.easts.baseline import cat_welf_east_baseline;
from project_easts.rollouts.cat_abc2a import cat_abc2a_east;

class cat_abc2a(cat_welf_east_baseline):
    MODEL_FN = cat_abc2a_east;
    PRFX = "abc2a";

    def model_training(this,inputs,DEV,bbone="densenet169"):
        ecfg = cat_config();
        ecfg.set("network", bbone);
        ecfg.set("weight_decay", 1e-5);
        ecfg.set("scope", this.PRFX);
        with tf.device(DEV):
            score,geometry,sel=this.MODEL_FN(ecfg).model([inputs[0]],True);

        cfg=this.LOSS_FN().get_default_config(with_instance_balance=1);

        dwd={};
        dwd["0xca39_scope"]=this.PRFX;

        loss_fn=this.LOSS_FN().init_ret(cfg,dwd);

        [loss_cls,loss_box,loss_ang]=loss_fn.call([inputs[1],score,inputs[2],geometry,inputs[3],inputs[4]],True);
        return [loss_cls,loss_box,loss_ang];

    def model_testing(this,inputs,network):
        cfg=cat_config();
        cfg.set("network",network);
        cfg.set("weight_decay",1e-5);
        cfg.set("scope",this.PRFX);
        score,geometry,sel=this.MODEL_FN(cfg).model(inputs,False);
        return score,geometry,sel;



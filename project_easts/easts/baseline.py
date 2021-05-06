import tensorflow as tf;

from slime.models.abstract_models import abstract_model;
from utils.cat_config import cat_config;
from slime.rollouts.libeast import welf_slime_east;
from generic.losses.cat_welf_east_losses import welf_east_ohem_loss_functor;

class cat_welf_east_baseline(abstract_model):
    MODEL_FN=welf_slime_east;
    PRFX="bl";
    LOSS_FN=welf_east_ohem_loss_functor;

    def model_training(this,inputs,DEV):
        ecfg = cat_config();
        ecfg.set("network", "densenet169");
        ecfg.set("weight_decay", 1e-5);
        ecfg.set("scope", this.PRFX);
        with tf.device(DEV):
            score,geometry=this.MODEL_FN(ecfg).model([inputs[0]],True);

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
        score,geometry=this.MODEL_FN(cfg).model(inputs,False);
        return score,geometry;


from neko_dogoo_v3.welf_east_losses import welf_ohem_loss_functor;
class cat_welf_east_baseline_ol(cat_welf_east_baseline):
    LOSS_FN=welf_ohem_loss_functor;

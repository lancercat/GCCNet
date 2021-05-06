import tensorflow as tf;
import numpy as np;


from generic.losses.cat_welf_east_losses import welf_tf_loss_helper;
from generic.losses.cat_welf_east_losses import welf_east_ohem_loss_functor;
from generic.losses.cat_welf_east_losses import WELF_CONSTS;
from utils.cat_config import cat_config;


class cat_welf_east_uni_ohem_loss_functor_kai(welf_east_ohem_loss_functor):

    @staticmethod
    def uni_weight_boosting_core(y_pred_cls,iou):
        cg=((y_pred_cls-0.8).clip(min=0)*5+1.)/2;
        rg=((iou-0.5).clip(min=0)*2+1.)/2;
        return cg,rg;
    def uni_weight_boosting(this,y_pred_cls,iou,classification_loss_n,box_loss_iou,box_loss_angle):
        cg,rg=tf.py_func(this.uni_weight_boosting_core,[y_pred_cls,iou],[tf.float32,tf.float32],False);
        return classification_loss_n*rg,cg*box_loss_iou, cg* box_loss_angle;

    def build_loss(this, y_true_cls, y_pred_cls,
                   y_true_geo, y_pred_geo,
                   training_mask, poly_mask,
                   with_ohem, with_instance_balance):

        classification_loss_n = welf_tf_loss_helper.cross_entropy_loss_with_mask(y_pred_cls, y_true_cls, training_mask)
        bmask = tf.cast(training_mask > 0, tf.float32);
        poly_val_mask = y_true_cls * bmask;
        box_loss_iou, box_loss_angle, iou = welf_tf_loss_helper.rbox_iou_loss(y_true_geo, y_pred_geo, poly_val_mask);

        classification_loss_n,box_loss_iou, box_loss_angle=this.uni_weight_boosting(y_pred_cls,iou,
                                                                                    classification_loss_n,box_loss_iou,box_loss_angle);

        cls_loss_w, box_loss_w, angle_loss_w = this.get_weights(classification_loss_n, box_loss_iou, box_loss_angle,
                                                                y_true_cls, poly_mask * bmask,
                                                                with_ohem, with_instance_balance
                                                                );

        classification_loss = tf.reduce_sum(classification_loss_n * cls_loss_w) / (
                tf.reduce_sum(cls_loss_w) + WELF_CONSTS.eps)
        box_loss = tf.reduce_sum(box_loss_iou * box_loss_w) / (tf.reduce_sum(box_loss_w) + WELF_CONSTS.eps)
        angle_loss = tf.reduce_sum(box_loss_angle * angle_loss_w) / (tf.reduce_sum(angle_loss_w) + WELF_CONSTS.eps)

        tf.summary.scalar('classification_loss', classification_loss)
        tf.summary.scalar('geometry_AABB', box_loss)
        tf.summary.scalar('geometry_theta', angle_loss)

        return [classification_loss, box_loss, angle_loss]

class cat_welf_east_uni_ohem_loss_functor_kai_ws(welf_east_ohem_loss_functor):

    @staticmethod
    def uni_weight_boosting_core(y_pred_cls,iou):
        cg=((y_pred_cls-0.8).clip(min=0)*5+1.)/2;
        rg=((iou-0.5).clip(min=0)*2+1.)/2;
        rg += (iou - 0.2).clip(max=0)  + 1.;
        return cg,rg;
    def uni_weight_boosting(this,y_pred_cls,iou,classification_loss_n,box_loss_iou,box_loss_angle):
        cg,rg=tf.py_func(this.uni_weight_boosting_core,[y_pred_cls,iou],[tf.float32,tf.float32],False);
        return classification_loss_n*rg,cg*box_loss_iou, cg* box_loss_angle;

    def build_loss(this, y_true_cls, y_pred_cls,
                   y_true_geo, y_pred_geo,
                   training_mask, poly_mask,
                   with_ohem, with_instance_balance):

        classification_loss_n = welf_tf_loss_helper.cross_entropy_loss_with_mask(y_pred_cls, y_true_cls, training_mask)
        bmask = tf.cast(training_mask > 0, tf.float32);
        poly_val_mask = y_true_cls * bmask;
        box_loss_iou, box_loss_angle, iou = welf_tf_loss_helper.rbox_iou_loss(y_true_geo, y_pred_geo, poly_val_mask);

        classification_loss_n,box_loss_iou, box_loss_angle=this.uni_weight_boosting(y_pred_cls,iou,
                                                                                    classification_loss_n,box_loss_iou,box_loss_angle);

        cls_loss_w, box_loss_w, angle_loss_w = this.get_weights(classification_loss_n, box_loss_iou, box_loss_angle,
                                                                y_true_cls, poly_mask * bmask,
                                                                with_ohem, with_instance_balance
                                                                );

        classification_loss = tf.reduce_sum(classification_loss_n * cls_loss_w) / (
                tf.reduce_sum(cls_loss_w) + WELF_CONSTS.eps)
        box_loss = tf.reduce_sum(box_loss_iou * box_loss_w) / (tf.reduce_sum(box_loss_w) + WELF_CONSTS.eps)
        angle_loss = tf.reduce_sum(box_loss_angle * angle_loss_w) / (tf.reduce_sum(angle_loss_w) + WELF_CONSTS.eps)

        tf.summary.scalar('classification_loss', classification_loss)
        tf.summary.scalar('geometry_AABB', box_loss)
        tf.summary.scalar('geometry_theta', angle_loss)

        return [classification_loss, box_loss, angle_loss]
# OHEM according to raw loss.
class cat_welf_east_uni_ohem_loss_functor_kaiII(welf_east_ohem_loss_functor):
    @staticmethod
    def uni_weight_boosting_core(y_pred_cls,iou):
        cg = ((y_pred_cls - 0.8).clip(min=0) * 5 + 1.) / 2;
        rg = ((iou - 0.5).clip(min=0) * 2 + 1.) / 2;
        return cg,rg;
    def uni_weight_boosting(this,y_pred_cls,iou,classification_loss_n,box_loss_iou,box_loss_angle):
        cg,rg=tf.py_func(type(this).uni_weight_boosting_core,[y_pred_cls,iou],[tf.float32,tf.float32],False);
        return classification_loss_n*rg,cg*box_loss_iou, cg* box_loss_angle;

    def build_loss(this, y_true_cls, y_pred_cls,
                   y_true_geo, y_pred_geo,
                   training_mask, poly_mask,
                   with_ohem, with_instance_balance):

        classification_loss_n = welf_tf_loss_helper.cross_entropy_loss_with_mask(y_pred_cls, y_true_cls, training_mask)
        bmask = tf.cast(training_mask > 0, tf.float32);
        poly_val_mask = y_true_cls * bmask;
        box_loss_iou, box_loss_angle, iou = welf_tf_loss_helper.rbox_iou_loss(y_true_geo, y_pred_geo, poly_val_mask);


        cls_loss_w, box_loss_w, angle_loss_w = this.get_weights(classification_loss_n, box_loss_iou, box_loss_angle,
                                                                y_true_cls, poly_mask * bmask,
                                                                with_ohem, with_instance_balance
                                                                );
        cls_loss_w, box_loss_w, angle_loss_w =this.uni_weight_boosting(y_pred_cls,iou,
                                                                                    cls_loss_w,box_loss_w,angle_loss_w);

        classification_loss = tf.reduce_sum(classification_loss_n * cls_loss_w) / (
                tf.reduce_sum(cls_loss_w) + WELF_CONSTS.eps)
        box_loss = tf.reduce_sum(box_loss_iou * box_loss_w) / (tf.reduce_sum(box_loss_w) + WELF_CONSTS.eps)
        angle_loss = tf.reduce_sum(box_loss_angle * angle_loss_w) / (tf.reduce_sum(angle_loss_w) + WELF_CONSTS.eps)

        tf.summary.scalar('classification_loss', classification_loss)
        tf.summary.scalar('geometry_AABB', box_loss)
        tf.summary.scalar('geometry_theta', angle_loss)

        return [classification_loss, box_loss, angle_loss];

class cat_welf_east_uni_ohem_loss_functor_kaif(welf_east_ohem_loss_functor):

    @staticmethod
    def uni_weight_boosting_core(y_pred_cls,iou,pos_mask):
        cg=((y_pred_cls-0.8).clip(min=0)*5+1.)/2;
        rg=((iou-0.5).clip(min=0)*2+1.)/2;
        rg[pos_mask<0.1]=1;
        # This does not apply to negative samples.
        return cg,rg;
    def uni_weight_boosting(this,y_pred_cls,iou,classification_loss_n,box_loss_iou,box_loss_angle,pos_mask):
        cg,rg=tf.py_func(this.uni_weight_boosting_core,[y_pred_cls,iou,pos_mask],[tf.float32,tf.float32],False);
        return classification_loss_n*rg,cg*box_loss_iou, cg* box_loss_angle;

    def build_loss(this, y_true_cls, y_pred_cls,
                   y_true_geo, y_pred_geo,
                   training_mask, poly_mask,
                   with_ohem, with_instance_balance):

        classification_loss_n = welf_tf_loss_helper.cross_entropy_loss_with_mask(y_pred_cls, y_true_cls, training_mask)
        bmask = tf.cast(training_mask > 0, tf.float32);
        poly_val_mask = y_true_cls * bmask;
        box_loss_iou, box_loss_angle, iou = welf_tf_loss_helper.rbox_iou_loss(y_true_geo, y_pred_geo, poly_val_mask);

        classification_loss_n,box_loss_iou, box_loss_angle=this.uni_weight_boosting(y_pred_cls,iou,
                                                                                    classification_loss_n,box_loss_iou,box_loss_angle,poly_val_mask);

        cls_loss_w, box_loss_w, angle_loss_w = this.get_weights(classification_loss_n, box_loss_iou, box_loss_angle,
                                                                y_true_cls, poly_mask * bmask,
                                                                with_ohem, with_instance_balance
                                                                );

        classification_loss = tf.reduce_sum(classification_loss_n * cls_loss_w) / (
                tf.reduce_sum(cls_loss_w) + WELF_CONSTS.eps)
        box_loss = tf.reduce_sum(box_loss_iou * box_loss_w) / (tf.reduce_sum(box_loss_w) + WELF_CONSTS.eps)
        angle_loss = tf.reduce_sum(box_loss_angle * angle_loss_w) / (tf.reduce_sum(angle_loss_w) + WELF_CONSTS.eps)

        tf.summary.scalar('classification_loss', classification_loss)
        tf.summary.scalar('geometry_AABB', box_loss)
        tf.summary.scalar('geometry_theta', angle_loss)

        return [classification_loss, box_loss, angle_loss]


# Crap I forgot the loss weight should not affect negative samples.
# IoU small or large shall not apply to negative samples, negative sample
class cat_welf_east_uni_ohem_loss_functor_kaiIIf(welf_east_ohem_loss_functor):
    @staticmethod
    def uni_weight_boosting_core(y_pred_cls, iou ,pos_mask):
        cg = ((y_pred_cls - 0.8).clip(min=0) * 5 + 1.) / 2;
        rg = ((iou - 0.5).clip(min=0) * 2 + 1.) / 2;
        rg[pos_mask<0.1]=1;
        return cg, rg;

    def uni_weight_boosting(this, y_pred_cls, iou, classification_loss_n, box_loss_iou, box_loss_angle,pos_mask):
        cg, rg = tf.py_func(type(this).uni_weight_boosting_core,
                            [y_pred_cls, iou,pos_mask], [tf.float32, tf.float32], False);
        return classification_loss_n * rg, cg * box_loss_iou, cg * box_loss_angle;

    def build_loss(this, y_true_cls, y_pred_cls,
                   y_true_geo, y_pred_geo,
                   training_mask, poly_mask,
                   with_ohem, with_instance_balance):
        classification_loss_n = welf_tf_loss_helper.cross_entropy_loss_with_mask(y_pred_cls, y_true_cls, training_mask)
        bmask = tf.cast(training_mask > 0, tf.float32);
        poly_val_mask = y_true_cls * bmask;

        box_loss_iou, box_loss_angle, iou = welf_tf_loss_helper.rbox_iou_loss(y_true_geo, y_pred_geo, poly_val_mask);

        cls_loss_w, box_loss_w, angle_loss_w = this.get_weights(classification_loss_n, box_loss_iou, box_loss_angle,
                                                                y_true_cls, poly_mask * bmask,
                                                                with_ohem, with_instance_balance
                                                                );
        cls_loss_w, box_loss_w, angle_loss_w = this.uni_weight_boosting(y_pred_cls, iou,
                                                                        cls_loss_w, box_loss_w, angle_loss_w,poly_val_mask);

        classification_loss = tf.reduce_sum(classification_loss_n * cls_loss_w) / (
                tf.reduce_sum(cls_loss_w) + WELF_CONSTS.eps)
        box_loss = tf.reduce_sum(box_loss_iou * box_loss_w) / (tf.reduce_sum(box_loss_w) + WELF_CONSTS.eps)
        angle_loss = tf.reduce_sum(box_loss_angle * angle_loss_w) / (tf.reduce_sum(angle_loss_w) + WELF_CONSTS.eps)

        tf.summary.scalar('classification_loss', classification_loss)
        tf.summary.scalar('geometry_AABB', box_loss)
        tf.summary.scalar('geometry_theta', angle_loss)

        return [classification_loss, box_loss, angle_loss];

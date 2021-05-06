import numpy as np;
import tensorflow as tf;
from generic.losses.welf_basic_np_loss_helper import welf_basic_np_loss_helper;


class cat_Plink_np_loss_helper:
    @staticmethod
    def sample_pesudo_error(inds, cls_losses,LTH):
        #Xent for prob 075 on negsamples
        ii = np.where(cls_losses[inds]>LTH);
        i=inds[ii];
        return i;

    @staticmethod
    def sample_ohem_mix(label, neg_inds, pos_inds, cls_losses, hard_neg_samples, rand_neg_samples):
        ineg = welf_basic_np_loss_helper.ohem_batch_over_group(neg_inds, cls_losses, hard_neg_samples,
                                                               rand_neg_samples);
        ipos = pos_inds;
        weight = np.zeros_like(label);
        all_cls_inds = np.concatenate([ineg, ipos]);
        weight[all_cls_inds] = 1;
        return weight,ineg;
    @staticmethod
    def sample_ohem_cls(label,cls_inds, cls_losses, hard_samples, rand_samples):
        #indicies
        IS=[];
        for i in range(len(cls_inds)):
            IS.append(welf_basic_np_loss_helper.ohem_batch_over_group(cls_inds[i], cls_losses,hard_samples[i],
                                                               rand_samples[i]));
        weight = np.zeros_like(label);
        all_cls_inds = np.concatenate(IS);
        weight[all_cls_inds] = 1;
        return weight;

class cat_tf_loss_helper:
    @staticmethod
    def sx_loss_with_mask(pred, label, mask, name='SoftmaxXEntropy_loss_with_mask'):
        with tf.name_scope(name):
            cost = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=pred,
                labels=label) ;
        return cost * mask;

    @staticmethod
    def sx_loss(pred, label, name='SoftmaxXEntropy_loss'):
        with tf.name_scope(name):
            cost = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=pred,
                labels=label);

        return cost ;

    #            pos_inds = np.where(label[i, :] == 1)[0]


class welf_tf_loss_helper:
    @staticmethod
    def cross_entropy_loss_with_mask(pred, label, mask, name='cross_entropy_loss_with_mask'):
        with tf.name_scope('cross_entropy_loss_with_mask'):
            bmask=tf.cast(mask >0,tf.float32);
            z = pred * bmask;
            y = label * bmask;

            eps = 1e-12
            loss_pos = - (y * tf.log(z + eps))
            loss_neg = (1. - y) * tf.log(1. - z + eps)
        cost = tf.subtract(loss_pos, loss_neg, name=name)
        return cost * mask;

    @staticmethod
    def box_iou(pred, label):
        [d1_gt, d2_gt, d3_gt, d4_gt] = label;
        [d1_pred, d2_pred, d3_pred, d4_pred] = pred;
        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
        w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
        h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        iou_approx = (area_intersect + 1.0) / (area_union + 1.0);
        return iou_approx;

    @staticmethod
    def rbox_iou_loss(pred, label, mask):
        """

        :param pred:     y_pred_geo
        :param label:   y_true_geo
        :param mask:        y_true_cls * training_mask
        :return:
        """
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=label, num_or_size_splits=5, axis=3);
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=pred, num_or_size_splits=5, axis=-1);

        iou_approx = welf_tf_loss_helper.box_iou([d1_pred, d2_pred, d3_pred, d4_pred], [d1_gt, d2_gt, d3_gt, d4_gt]);

        L_AABB = -tf.log(iou_approx);
        L_theta = 1 - tf.cos(theta_pred - theta_gt)
        box_loss_angle = L_theta * mask;
        box_loss_iou = L_AABB * mask;
        iou_approx*=mask;
        return box_loss_iou,box_loss_angle, iou_approx;

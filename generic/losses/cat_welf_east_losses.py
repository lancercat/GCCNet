import tensorflow as tf;
import numpy as np;
from neko_dogoo_v3.dogoo import dogoo_functor;
from utils.cat_config import cat_config;
from generic.losses.welf_basic_np_loss_helper import welf_basic_np_loss_helper;

from generic.losses.welf_loss_utils import welf_tf_loss_helper;
from generic.losses.welf_basic_np_loss_helper import welf_EAST_np_loss_helper;
from generic.losses.welf_basic_np_loss_helper import WELF_CONSTS;

from functools import partial;


class welf_east_ohem_loss_functor(dogoo_functor):

    def __init__(this):
        super().__init__();

    def _init_call_back(this):
        this.with_ohem = this.config_tree.get(int, "with_ohem");
        this.with_instance_balance = this.config_tree.get(int, "with_instance_balance");
        this.rand_neg_samples = this.config_tree.get(float, "rand_neg_samples");
        this.hard_neg_samples = this.config_tree.get(float, "hard_neg_samples");
        this.rand_reg_samples = this.config_tree.get(float, "rand_reg_samples");
        this.hard_reg_samples = this.config_tree.get(float, "hard_reg_samples");

    def get_default_config(_, **args):
        config = cat_config();
        config.set("with_ohem", args.get("with_ohem", 1));
        config.set("with_instance_balance", args.get("with_instance_balance", 1));
        config.set("rand_neg_samples", args.get("rand_neg_samples", 1./32));
        config.set("hard_neg_samples", args.get("hard_neg_samples", 1./32));
        config.set("rand_reg_samples", args.get("rand_reg_samples", 1./128));
        config.set("hard_reg_samples", args.get("hard_reg_samples", 1./128));

        return config;
    def _make_weight_dict(_, config_tree, inputs, scope):
        return {}, _._infer(config_tree, {}, inputs);

    def call(this, inputs, is_training):
        return this.build_loss(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], this.with_ohem,
                               this.with_instance_balance);

    def get_weights(this, classification_loss, box_loss_iou, box_loss_angle,
                    y_true_cls, masked_poly_mask,
                    with_ohem, is_ib):
        cls_loss_w = tf.ones_like(classification_loss);
        box_loss_w = tf.ones_like(box_loss_iou);
        angle_loss_w = tf.ones_like(box_loss_angle);
        if (with_ohem):
            ohemfn = partial(welf_EAST_np_loss_helper.sample_ohem_EAST, hard_neg_samples=this.hard_neg_samples,
                             rand_neg_samples=this.rand_neg_samples,
                             hard_reg_samples=this.hard_reg_samples, rand_reg_samples=this.rand_reg_samples);
            cls_weights, box_weights, angle_weights = tf.py_func(ohemfn,
                                                                 [y_true_cls, classification_loss, box_loss_iou,
                                                                  box_loss_angle],
                                                                 [tf.float32, tf.float32, tf.float32]);
            cls_loss_w = cls_loss_w * cls_weights;
            box_loss_w = box_loss_w * box_weights;
            angle_loss_w = angle_loss_w * angle_weights;
        if (is_ib):
            ib_weights = tf.py_func(welf_basic_np_loss_helper.get_instance_balanced_weight, [masked_poly_mask], [tf.float32])[
                0]

            cls_loss_w = cls_loss_w * ib_weights;
            box_loss_w = box_loss_w * ib_weights;
            angle_loss_w = angle_loss_w * ib_weights;

        return cls_loss_w, box_loss_w, angle_loss_w;

    def build_loss(this, y_true_cls, y_pred_cls,
                   y_true_geo, y_pred_geo,
                   training_mask, poly_mask,
                   with_ohem, with_instance_balance):
        '''
        define the loss used for training, contraning 3 part,
        the first part we use [ohemed][ibed] logloss,
        the second part is the iou loss defined in the paper
        we put these together only because it's faster.
        :param y_true_cls: ground truth of text
        :param y_pred_cls: prediction os text
        :param y_true_geo: ground truth of geometry
        :param y_pred_geo: prediction of geometry
        :param training_mask: mask used in training, to ignore some text annotated by ###
        :return:
        '''

        classification_loss_n = welf_tf_loss_helper.cross_entropy_loss_with_mask(y_pred_cls, y_true_cls, training_mask)
        bmask = tf.cast(training_mask > 0, tf.float32);
        poly_val_mask = y_true_cls * bmask;
        box_loss_iou, box_loss_angle, _ = welf_tf_loss_helper.rbox_iou_loss(y_true_geo, y_pred_geo, poly_val_mask);

        cls_loss_w, box_loss_w, angle_loss_w = this.get_weights(classification_loss_n, box_loss_iou, box_loss_angle,
                                                                y_true_cls, poly_mask * bmask,
                                                                with_ohem, with_instance_balance
                                                                );

        classification_loss = tf.reduce_sum(classification_loss_n * cls_loss_w) / \
                              (tf.reduce_sum(cls_loss_w) + WELF_CONSTS.eps)
        box_loss = tf.reduce_sum(box_loss_iou * box_loss_w) / (tf.reduce_sum(box_loss_w) + WELF_CONSTS.eps)
        angle_loss = tf.reduce_sum(box_loss_angle * angle_loss_w) / (tf.reduce_sum(angle_loss_w) + WELF_CONSTS.eps)

        tf.summary.scalar('classification_loss', classification_loss)
        tf.summary.scalar('geometry_AABB', box_loss)
        tf.summary.scalar('geometry_theta', angle_loss)

        return [classification_loss, box_loss, angle_loss]


# #
#     def dice_coefficient(this, y_true_cls, y_pred_cls,
#                          training_mask):
#         '''
#         dice loss
#         :param y_true_cls:
#         :param y_pred_cls:
#         :param training_mask:
#         :return:
#         '''
#         eps = 1e-5
#         intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
#         union = tf.reduce_sum(y_true_cls * training_mask) + tf.reduce_sum(y_pred_cls * training_mask) + eps
#         loss = 1. - (2 * intersection / union + WELF_CONSTS.eps)
#         tf.summary.scalar('classification_dice_loss', loss)
#         return loss
#
#     def batch_flatten(this, x):
#         """
#         Flatten the tensor except the first dimension.
#         """
#         shape = x.get_shape().as_list()[1:]
#         if None not in shape:
#             return tf.reshape(x, [-1, int(np.prod(shape))])
#         return tf.reshape(x, tf.stack([tf.shape(x)[0], -1]))
#
#     def class_balanced_cross_entropy_no_norm(this, pred, label, name='cross_entropy_loss_no_norm'):
#         """
#         The class-balanced cross entropy loss,
#         as in `Holistically-Nested Edge Detection
#         <http://arxiv.org/abs/1504.06375>`_.
#         Args:
#             pred: of shape (b, ...). the predictions in [0,1].
#             label: of the same shape. the ground truth in {0,1}.
#         Returns:
#             class-balanced cross entropy loss.
#         """
#         z = this.batch_flatten(pred)
#         y = tf.cast(this.batch_flatten(label), tf.float32)
#
#         count_neg = tf.reduce_sum(1. - y)
#         count_pos = tf.reduce_sum(y)
#         beta = count_neg / (count_neg + count_pos + WELF_CONSTS.eps)
#
#         eps = 1e-12
#         loss_pos = -beta * tf.reduce_sum(y * tf.log(z + eps))
#         loss_neg = (1. - beta) * tf.reduce_sum((1. - y) * tf.log(1. - z + eps))
#         cost = tf.subtract(loss_pos, loss_neg, name=name) / (tf.cast(tf.shape(pred)[0], tf.float32) + WELF_CONSTS.eps)
#         return cost
#
#     def class_balanced_sigmoid_cross_entropy_no_norm(this, logits, label, name='cross_entropy_loss_no_norm'):
#         """
#         This function accepts logits rather than predictions, and is more numerically stable than
#         :func:`class_balanced_cross_entropy`.
#         """
#
#         y = tf.cast(label, tf.float32)
#
#         count_neg = tf.reduce_sum(1. - y)  # the number of 0 in y
#         count_pos = tf.reduce_sum(y)  # the number of 1 in y (less than count_neg)
#         beta = count_neg / ((count_neg + count_pos) + WELF_CONSTS.eps);
#
#         pos_weight = beta / ((1 - beta) + WELF_CONSTS.eps)
#         cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y, pos_weight=pos_weight)
#
#         cost = tf.reduce_sum(cost * (1 - beta), name=name) / (
#                     tf.cast(tf.shape(logits)[0], tf.float32) + WELF_CONSTS.eps)
#         return cost
#
#     def class_balanced_cross_entropy(this, pred, label, name='cross_entropy_loss'):
#         """
#         The class-balanced cross entropy loss,
#         as in `Holistically-Nested Edge Detection
#         <http://arxiv.org/abs/1504.06375>`_.
#
#         Args:
#             pred: of shape (b, ...). the predictions in [0,1].
#             label: of the same shape. the ground truth in {0,1}.
#         Returns:
#             class-balanced cross entropy loss.
#         """
#         with tf.name_scope('class_balanced_cross_entropy'):
#             z = this.batch_flatten(pred)
#             y = tf.cast(this.batch_flatten(label), tf.float32)
#
#             count_neg = tf.reduce_sum(1. - y)
#             count_pos = tf.reduce_sum(y)
#             beta = count_neg / ((count_neg + count_pos) + WELF_CONSTS.eps)
#
#             eps = 1e-12
#             loss_pos = -beta * tf.reduce_mean(y * tf.log(z + eps))
#             loss_neg = (1. - beta) * tf.reduce_mean((1. - y) * tf.log(1. - z + eps))
#         cost = tf.subtract(loss_pos, loss_neg, name=name)
#         return cost
#
#     def class_balanced_sigmoid_cross_entropy(this, logits, label, name='cross_entropy_loss'):
#         """
#         This function accepts logits rather than predictions, and is more numerically stable than
#         :func:`class_balanced_cross_entropy`.
#         """
#         with tf.name_scope('class_balanced_sigmoid_cross_entropy'):
#             y = tf.cast(label, tf.float32)
#
#             count_neg = tf.reduce_sum(1. - y)
#             count_pos = tf.reduce_sum(y)
#             beta = count_neg / ((count_neg + count_pos) + WELF_CONSTS.eps)
#
#             pos_weight = beta / ((1 - beta) + WELF_CONSTS.eps)
#             cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y, pos_weight=pos_weight)
#             cost = tf.reduce_mean(cost * (1 - beta))
#             zero = tf.equal(count_pos, 0.0)
#         return tf.where(zero, 0.0, cost, name=name)
#
#
#

import tensorflow as tf;
import numpy as np;
from neko_dogoo_v3.dogoo import dogoo_functor;
from utils.cat_config import cat_config;

class welf_ohem_loss_functor(dogoo_functor):
    EPS=0.00000009;
    def __init__(this):
        super().__init__();
    def _init_call_back(this):
        this.with_ohem = this.config_tree.get(int,"with_ohem");
        this.with_instance_balance= this.config_tree.get(int,"with_instance_balance");
        this.rand_neg_samples =  this.config_tree.get(int,"rand_neg_samples");
        this.hard_neg_samples =  this.config_tree.get(int,"hard_neg_samples");
        this.rand_reg_samples =  this.config_tree.get(int,"rand_reg_samples");
        this.hard_reg_samples = this.config_tree.get(int,"hard_reg_samples");

    def get_default_config(_,**args):
        config=cat_config();
        config.set("with_ohem",args.get("with_ohem",1));
        config.set("with_instance_balance", args.get("with_instance_balance",1));
        config.set("rand_neg_samples",args.get("rand_neg_samples",128));
        config.set("hard_neg_samples", args.get("rand_neg_samples", 128));
        config.set("rand_reg_samples", args.get("rand_reg_samples", 128));
        config.set("hard_reg_samples", args.get("rand_reg_samples", 128));

        return config;

    def _make_weight_dict(_, config_tree, inputs, scope):
        return {},_._infer(config_tree,{},inputs);


    def call(this, inputs,is_training):
        return this.build_loss(inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],inputs[5],this.with_ohem,this.with_instance_balance);


    def dice_coefficient(this,y_true_cls, y_pred_cls,
                         training_mask):
        '''
        dice loss
        :param y_true_cls:
        :param y_pred_cls:
        :param training_mask:
        :return:
        '''
        eps = 1e-5
        intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
        union = tf.reduce_sum(y_true_cls * training_mask) + tf.reduce_sum(y_pred_cls * training_mask) + eps
        loss = 1. - (2 * intersection / (union+this.EPS))
        tf.summary.scalar('classification_dice_loss', loss)
        return loss

    def batch_flatten(this,x):
        """
        Flatten the tensor except the first dimension.
        """
        shape = x.get_shape().as_list()[1:]
        if None not in shape:
            return tf.reshape(x, [-1, int(np.prod(shape))])
        return tf.reshape(x, tf.stack([tf.shape(x)[0], -1]))

    def class_balanced_cross_entropy_no_norm(this,pred, label, name='cross_entropy_loss_no_norm'):
        """
        The class-balanced cross entropy loss,
        as in `Holistically-Nested Edge Detection
        <http://arxiv.org/abs/1504.06375>`_.
        Args:
            pred: of shape (b, ...). the predictions in [0,1].
            label: of the same shape. the ground truth in {0,1}.
        Returns:
            class-balanced cross entropy loss.
        """
        z = this.batch_flatten(pred)
        y = tf.cast(this.batch_flatten(label), tf.float32)

        count_neg = tf.reduce_sum(1. - y)
        count_pos = tf.reduce_sum(y)
        beta = count_neg / (count_neg + count_pos+this.EPS)

        eps = 1e-12
        loss_pos = -beta * tf.reduce_sum(y * tf.log(z + eps))
        loss_neg = (1. - beta) * tf.reduce_sum((1. - y) * tf.log(1. - z + eps))
        cost = tf.subtract(loss_pos, loss_neg, name=name) / (tf.cast(tf.shape(pred)[0], tf.float32)+this.EPS)
        return cost

    def class_balanced_sigmoid_cross_entropy_no_norm(this,logits, label, name='cross_entropy_loss_no_norm'):
        """
        This function accepts logits rather than predictions, and is more numerically stable than
        :func:`class_balanced_cross_entropy`.
        """

        y = tf.cast(label, tf.float32)

        count_neg = tf.reduce_sum(1. - y)  # the number of 0 in y
        count_pos = tf.reduce_sum(y)  # the number of 1 in y (less than count_neg)
        beta = count_neg / ((count_neg + count_pos)+this.EPS);

        pos_weight = beta / ((1 - beta)+this.EPS)
        cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y, pos_weight=pos_weight)

        cost = tf.reduce_sum(cost * (1 - beta), name=name) / (tf.cast(tf.shape(logits)[0], tf.float32)+this.EPS)
        return cost

    def class_balanced_cross_entropy(this,pred, label, name='cross_entropy_loss'):
        """
        The class-balanced cross entropy loss,
        as in `Holistically-Nested Edge Detection
        <http://arxiv.org/abs/1504.06375>`_.

        Args:
            pred: of shape (b, ...). the predictions in [0,1].
            label: of the same shape. the ground truth in {0,1}.
        Returns:
            class-balanced cross entropy loss.
        """
        with tf.name_scope('class_balanced_cross_entropy'):
            z = this.batch_flatten(pred)
            y = tf.cast(this.batch_flatten(label), tf.float32)

            count_neg = tf.reduce_sum(1. - y)
            count_pos = tf.reduce_sum(y)
            beta = count_neg / ((count_neg + count_pos)+this.EPS)

            eps = 1e-12
            loss_pos = -beta * tf.reduce_mean(y * tf.log(z + eps))
            loss_neg = (1. - beta) * tf.reduce_mean((1. - y) * tf.log(1. - z + eps))
        cost = tf.subtract(loss_pos, loss_neg, name=name)
        return cost

    def class_balanced_sigmoid_cross_entropy(this,logits, label, name='cross_entropy_loss'):
        """
        This function accepts logits rather than predictions, and is more numerically stable than
        :func:`class_balanced_cross_entropy`.
        """
        with tf.name_scope('class_balanced_sigmoid_cross_entropy'):
            y = tf.cast(label, tf.float32)

            count_neg = tf.reduce_sum(1. - y)
            count_pos = tf.reduce_sum(y)
            beta = count_neg / ((count_neg + count_pos)+this.EPS)

            pos_weight = beta / ((1 - beta)+this.EPS)
            cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y, pos_weight=pos_weight)
            cost = tf.reduce_mean(cost * (1 - beta))
            zero = tf.equal(count_pos, 0.0)
        return tf.where(zero, 0.0, cost, name=name)

    def cross_entropy_loss_with_mask(this,pred, label, mask, name='cross_entropy_loss_with_mask'):
        with tf.name_scope('cross_entropy_loss_with_mask'):
            bmask=tf.cast(mask >0,tf.float32);
            z = pred * bmask;
            y = label * bmask;

            eps = 1e-12
            loss_pos = - (y * tf.log(z + eps))
            loss_neg = (1. - y) * tf.log(1. - z + eps)
        cost = tf.subtract(loss_pos, loss_neg, name=name)
        return cost * mask;

    def loss(this,y_true_cls, y_pred_cls,
             y_true_geo, y_pred_geo,
             training_mask):
        '''
        define the loss used for training, contraning two part,
        the first part we use dice loss instead of weighted logloss,
        the second part is the iou loss defined in the paper
        :param y_true_cls: ground truth of text
        :param y_pred_cls: prediction os text
        :param y_true_geo: ground truth of geometry
        :param y_pred_geo: prediction of geometry
        :param training_mask: mask used in training, to ignore some text annotated by ###
        :return:
        '''
        classification_loss = this.dice_coefficient(y_true_cls, y_pred_cls, training_mask)
        # scale classification loss to match the iou loss part
        classification_loss *= 0.01

        # d1 -> top, d2->right, d3->bottom, d4->left
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)
        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
        w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
        h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        L_AABB = -tf.log((area_intersect + 1.0) / (area_union + 1.0))
        L_theta = 1 - tf.cos(theta_pred - theta_gt)
        tf.summary.scalar('geometry_AABB', tf.reduce_mean(L_AABB * y_true_cls * training_mask))
        tf.summary.scalar('geometry_theta', tf.reduce_mean(L_theta * y_true_cls * training_mask))
        L_g = L_AABB + 20 * L_theta

        return tf.reduce_mean(L_g * y_true_cls * training_mask) + classification_loss

    def sample_ohem(this,y_true_cls, cls_loss, box_loss, angle_loss,
                    ):

        label = y_true_cls.reshape((y_true_cls.shape[0], -1))
        cls = cls_loss.reshape((cls_loss.shape[0], -1))
        box = box_loss.reshape((box_loss.shape[0], -1))
        angle = angle_loss.reshape((angle_loss.shape[0], -1))
        cls_label_inds = np.zeros_like(label)
        box_label_inds = np.zeros_like(label)
        angle_label_inds = np.zeros_like(label)
        for i in range(y_true_cls.shape[0]):
            # for cls
            pos_inds = np.where(label[i, :] == 1)[0]
            neg_inds = np.where(label[i, :] == 0)[0]
            neg_sorted_inds = np.argsort(cls[i, [neg_inds]])[0]
            k_cls_neg_hard = np.min([neg_inds.shape[0], this.hard_neg_samples])
            cls_neg_hard_indices = neg_inds[neg_sorted_inds][-k_cls_neg_hard:]
            k_cls_neg_rand = np.max([0, np.min([this.rand_neg_samples, neg_inds.shape[0] - k_cls_neg_hard])])
            if k_cls_neg_rand > 0:
                cls_neg_rand_indices = np.random.choice(neg_inds[neg_sorted_inds][:-k_cls_neg_hard],
                                                        size=k_cls_neg_rand, replace=False)
            else:
                cls_neg_rand_indices = np.array([], dtype=np.int)
            cls_inds = np.concatenate([pos_inds, cls_neg_hard_indices, cls_neg_rand_indices])
            cls_label_inds[i, cls_inds] = 1
            # for box
            pos_sorted_inds = np.argsort(box[i, [pos_inds]])[0]
            k_box_pos_hard = np.min([this.hard_reg_samples, pos_inds.shape[0]])
            box_pos_hard_indices = pos_inds[pos_sorted_inds][:k_box_pos_hard]
            k_box_pos_rand = np.max([0, np.min([this.rand_reg_samples, pos_inds.shape[0] - k_box_pos_hard]), 0])
            if k_box_pos_rand > 0:
                box_pos_rand_indices = np.random.choice(pos_inds[pos_sorted_inds][k_box_pos_hard:], size=k_box_pos_rand,
                                                        replace=False)
            else:
                box_pos_rand_indices = np.array([], dtype=np.int)
            box_inds = np.concatenate([box_pos_hard_indices, box_pos_rand_indices])
            box_label_inds[i, box_inds] = 1
            # for angle
            pos_sorted_inds = np.argsort(angle[i, [pos_inds]])[0]
            k_angle_pos_hard = np.min([this.hard_reg_samples, pos_inds.shape[0]])
            angle_pos_hard_indices = pos_inds[pos_sorted_inds][:k_angle_pos_hard]
            k_angle_pos_rand = np.max([0, np.min([this.rand_reg_samples, pos_inds.shape[0] - k_angle_pos_hard]), 0])
            if k_angle_pos_rand > 0:
                angle_pos_rand_indices = np.random.choice(pos_inds[pos_sorted_inds][k_angle_pos_hard:],
                                                          size=k_angle_pos_rand, replace=False)
            else:
                angle_pos_rand_indices = np.array([], dtype=np.int)
            angle_inds = np.concatenate([angle_pos_hard_indices, angle_pos_rand_indices])
            angle_label_inds[i, angle_inds] = 1
        return cls_label_inds.reshape(y_true_cls.shape), box_label_inds.reshape(
            y_true_cls.shape), angle_label_inds.reshape(y_true_cls.shape)

    def get_instance_balanced_weight(this,poly_mask):
        ib_weight = np.ones_like(poly_mask, dtype=np.float32)
        for n in range(poly_mask.shape[0]):
            total_area = np.where(poly_mask[n] != 0)[0].shape[0]
            poly_num = int(np.max(poly_mask[n]))
            poly_area_list = []
            for i in range(poly_num):
                poly_i = np.where(poly_mask[n] == i + 1, np.ones_like(poly_mask[n]), np.zeros_like(poly_mask[n]))
                area_i = np.sum(poly_i)
                if area_i == 0:
                    continue
                else:
                    poly_area_list.append((i + 1, area_i))
            # print(len(poly_area_list))
            if len(poly_area_list) == 0:
                continue
            B = float(total_area) / (len(poly_area_list)+this.EPS)
            for p, area_p in poly_area_list:
                ib_weight[n, :, :, :] = np.where(poly_mask[n] == p,
                                                 np.ones_like(poly_mask[n], dtype=np.float32) * B / (area_p+this.EPS),
                                                 ib_weight[n])
        return ib_weight

    def build_loss(this,y_true_cls, y_pred_cls,
                   y_true_geo, y_pred_geo,
                   training_mask, poly_mask,
                   with_ohem, with_instance_balance):
        '''
        define the loss used for training, contraning two part,
        the first part we use dice loss instead of weighted logloss,
        the second part is the iou loss defined in the paper
        :param y_true_cls: ground truth of text
        :param y_pred_cls: prediction os text
        :param y_true_geo: ground truth of geometry
        :param y_pred_geo: prediction of geometry
        :param training_mask: mask used in training, to ignore some text annotated by ###
        :return:
        '''

        classification_loss_n = this.cross_entropy_loss_with_mask(y_pred_cls, y_true_cls, training_mask)

        # d1 -> top, d2->right, d3->bottom, d4->left
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3,name="aho")
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=-1,name="bokeh");
        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
        w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
        h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        L_AABB = -tf.log((area_intersect + 1.0) / (area_union + 1.0))
        L_theta = 1 - tf.cos(theta_pred - theta_gt)
        box_loss_angle = L_theta * y_true_cls * training_mask
        box_loss_iou = L_AABB * y_true_cls * training_mask

        if with_ohem:
            cls_indices, box_indices, angle_indices = tf.py_func(this.sample_ohem,
                                                                 [y_true_cls, classification_loss_n, box_loss_iou,
                                                                  box_loss_angle], [tf.float32, tf.float32, tf.float32])
            cls_indices.set_shape([None, None, None, 1])
            box_indices.set_shape([None, None, None, 1])
            angle_indices.set_shape([None, None, None, 1])
            if with_instance_balance:
                ib_weights = tf.py_func(this.get_instance_balanced_weight, [poly_mask * training_mask], [tf.float32])[0]
                ib_weights.set_shape([None, None, None, 1])
                cls_indices *= ib_weights
                box_indices *= ib_weights
                angle_indices *= ib_weights
            classification_loss = tf.reduce_sum(classification_loss_n * cls_indices) / (tf.reduce_sum(cls_indices)+this.EPS)
            box_loss = tf.reduce_sum(box_loss_iou * box_indices) / (tf.reduce_sum(box_indices)+this.EPS)
            angle_loss = tf.reduce_sum(box_loss_angle * angle_indices) / (tf.reduce_sum(angle_indices)+this.EPS)
            tf.summary.scalar('classification_loss', classification_loss)
            tf.summary.scalar('geometry_AABB', box_loss)
            tf.summary.scalar('geometry_theta', angle_loss)
        else:
            if with_instance_balance:
                ib_weights = tf.py_func(this.get_instance_balanced_weight, [poly_mask], [tf.float32])[0]
                ib_weights.set_shape([None, None, None, 1])
                classification_loss = tf.reduce_mean(classification_loss_n * ib_weights)
                box_loss = tf.reduce_mean(box_loss_iou * ib_weights)
                angle_loss = tf.reduce_mean(box_loss_angle * ib_weights)
            else:
                classification_loss = tf.reduce_mean(classification_loss_n)
                box_loss = tf.reduce_mean(box_loss_iou)
                angle_loss = tf.reduce_mean(box_loss_angle)
            tf.summary.scalar('classification_loss', classification_loss)
            tf.summary.scalar('geometry_AABB', box_loss)
            tf.summary.scalar('geometry_theta', angle_loss)

        return [classification_loss , box_loss , angle_loss]



#
import tensorflow as tf;
import numpy as np;
from neko_dogoo_v3.dogoo import dogoo_functor;
from utils.cat_config import cat_config;

class welf_ohem_loss_functor(dogoo_functor):
    EPS=0.00000009;
    def __init__(this):
        super().__init__();
    def _init_call_back(this):
        this.with_ohem = this.config_tree.get(int,"with_ohem");
        this.with_instance_balance= this.config_tree.get(int,"with_instance_balance");
        this.rand_neg_samples =  this.config_tree.get(int,"rand_neg_samples");
        this.hard_neg_samples =  this.config_tree.get(int,"hard_neg_samples");
        this.rand_reg_samples =  this.config_tree.get(int,"rand_reg_samples");
        this.hard_reg_samples = this.config_tree.get(int,"hard_reg_samples");

    def get_default_config(_,**args):
        config=cat_config();
        config.set("with_ohem",args.get("with_ohem",1));
        config.set("with_instance_balance", args.get("with_instance_balance",1));
        config.set("rand_neg_samples",args.get("rand_neg_samples",128));
        config.set("hard_neg_samples", args.get("rand_neg_samples", 128));
        config.set("rand_reg_samples", args.get("rand_reg_samples", 128));
        config.set("hard_reg_samples", args.get("rand_reg_samples", 128));

        return config;

    def _make_weight_dict(_, config_tree, inputs, scope):
        return {},_._infer(config_tree,{},inputs);


    def call(this, inputs,is_training):
        return this.build_loss(inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],inputs[5],this.with_ohem,this.with_instance_balance);


    def dice_coefficient(this,y_true_cls, y_pred_cls,
                         training_mask):
        '''
        dice loss
        :param y_true_cls:
        :param y_pred_cls:
        :param training_mask:
        :return:
        '''
        eps = 1e-5
        intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
        union = tf.reduce_sum(y_true_cls * training_mask) + tf.reduce_sum(y_pred_cls * training_mask) + eps
        loss = 1. - (2 * intersection / (union+this.EPS))
        tf.summary.scalar('classification_dice_loss', loss)
        return loss

    def batch_flatten(this,x):
        """
        Flatten the tensor except the first dimension.
        """
        shape = x.get_shape().as_list()[1:]
        if None not in shape:
            return tf.reshape(x, [-1, int(np.prod(shape))])
        return tf.reshape(x, tf.stack([tf.shape(x)[0], -1]))

    def class_balanced_cross_entropy_no_norm(this,pred, label, name='cross_entropy_loss_no_norm'):
        """
        The class-balanced cross entropy loss,
        as in `Holistically-Nested Edge Detection
        <http://arxiv.org/abs/1504.06375>`_.
        Args:
            pred: of shape (b, ...). the predictions in [0,1].
            label: of the same shape. the ground truth in {0,1}.
        Returns:
            class-balanced cross entropy loss.
        """
        z = this.batch_flatten(pred)
        y = tf.cast(this.batch_flatten(label), tf.float32)

        count_neg = tf.reduce_sum(1. - y)
        count_pos = tf.reduce_sum(y)
        beta = count_neg / (count_neg + count_pos+this.EPS)

        eps = 1e-12
        loss_pos = -beta * tf.reduce_sum(y * tf.log(z + eps))
        loss_neg = (1. - beta) * tf.reduce_sum((1. - y) * tf.log(1. - z + eps))
        cost = tf.subtract(loss_pos, loss_neg, name=name) / (tf.cast(tf.shape(pred)[0], tf.float32)+this.EPS)
        return cost

    def class_balanced_sigmoid_cross_entropy_no_norm(this,logits, label, name='cross_entropy_loss_no_norm'):
        """
        This function accepts logits rather than predictions, and is more numerically stable than
        :func:`class_balanced_cross_entropy`.
        """

        y = tf.cast(label, tf.float32)

        count_neg = tf.reduce_sum(1. - y)  # the number of 0 in y
        count_pos = tf.reduce_sum(y)  # the number of 1 in y (less than count_neg)
        beta = count_neg / ((count_neg + count_pos)+this.EPS);

        pos_weight = beta / ((1 - beta)+this.EPS)
        cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y, pos_weight=pos_weight)

        cost = tf.reduce_sum(cost * (1 - beta), name=name) / (tf.cast(tf.shape(logits)[0], tf.float32)+this.EPS)
        return cost

    def class_balanced_cross_entropy(this,pred, label, name='cross_entropy_loss'):
        """
        The class-balanced cross entropy loss,
        as in `Holistically-Nested Edge Detection
        <http://arxiv.org/abs/1504.06375>`_.

        Args:
            pred: of shape (b, ...). the predictions in [0,1].
            label: of the same shape. the ground truth in {0,1}.
        Returns:
            class-balanced cross entropy loss.
        """
        with tf.name_scope('class_balanced_cross_entropy'):
            z = this.batch_flatten(pred)
            y = tf.cast(this.batch_flatten(label), tf.float32)

            count_neg = tf.reduce_sum(1. - y)
            count_pos = tf.reduce_sum(y)
            beta = count_neg / ((count_neg + count_pos)+this.EPS)

            eps = 1e-12
            loss_pos = -beta * tf.reduce_mean(y * tf.log(z + eps))
            loss_neg = (1. - beta) * tf.reduce_mean((1. - y) * tf.log(1. - z + eps))
        cost = tf.subtract(loss_pos, loss_neg, name=name)
        return cost

    def class_balanced_sigmoid_cross_entropy(this,logits, label, name='cross_entropy_loss'):
        """
        This function accepts logits rather than predictions, and is more numerically stable than
        :func:`class_balanced_cross_entropy`.
        """
        with tf.name_scope('class_balanced_sigmoid_cross_entropy'):
            y = tf.cast(label, tf.float32)

            count_neg = tf.reduce_sum(1. - y)
            count_pos = tf.reduce_sum(y)
            beta = count_neg / ((count_neg + count_pos)+this.EPS)

            pos_weight = beta / ((1 - beta)+this.EPS)
            cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y, pos_weight=pos_weight)
            cost = tf.reduce_mean(cost * (1 - beta))
            zero = tf.equal(count_pos, 0.0)
        return tf.where(zero, 0.0, cost, name=name)

    def cross_entropy_loss_with_mask(this,pred, label, mask, name='cross_entropy_loss_with_mask'):
        with tf.name_scope('cross_entropy_loss_with_mask'):
            bmask=tf.cast(mask >0,tf.float32);
            z = pred * bmask;
            y = label * bmask;

            eps = 1e-12
            loss_pos = - (y * tf.log(z + eps))
            loss_neg = (1. - y) * tf.log(1. - z + eps)
        cost = tf.subtract(loss_pos, loss_neg, name=name)
        return cost * mask;

    def loss(this,y_true_cls, y_pred_cls,
             y_true_geo, y_pred_geo,
             training_mask):
        '''
        define the loss used for training, contraning two part,
        the first part we use dice loss instead of weighted logloss,
        the second part is the iou loss defined in the paper
        :param y_true_cls: ground truth of text
        :param y_pred_cls: prediction os text
        :param y_true_geo: ground truth of geometry
        :param y_pred_geo: prediction of geometry
        :param training_mask: mask used in training, to ignore some text annotated by ###
        :return:
        '''
        classification_loss = this.dice_coefficient(y_true_cls, y_pred_cls, training_mask)
        # scale classification loss to match the iou loss part
        classification_loss *= 0.01

        # d1 -> top, d2->right, d3->bottom, d4->left
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)
        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
        w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
        h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        L_AABB = -tf.log((area_intersect + 1.0) / (area_union + 1.0))
        L_theta = 1 - tf.cos(theta_pred - theta_gt)
        tf.summary.scalar('geometry_AABB', tf.reduce_mean(L_AABB * y_true_cls * training_mask))
        tf.summary.scalar('geometry_theta', tf.reduce_mean(L_theta * y_true_cls * training_mask))
        L_g = L_AABB + 20 * L_theta

        return tf.reduce_mean(L_g * y_true_cls * training_mask) + classification_loss

    def sample_ohem(this,y_true_cls, cls_loss, box_loss, angle_loss,
                    ):

        label = y_true_cls.reshape((y_true_cls.shape[0], -1))
        cls = cls_loss.reshape((cls_loss.shape[0], -1))
        box = box_loss.reshape((box_loss.shape[0], -1))
        angle = angle_loss.reshape((angle_loss.shape[0], -1))
        cls_label_inds = np.zeros_like(label)
        box_label_inds = np.zeros_like(label)
        angle_label_inds = np.zeros_like(label)
        for i in range(y_true_cls.shape[0]):
            # for cls
            pos_inds = np.where(label[i, :] == 1)[0]
            neg_inds = np.where(label[i, :] == 0)[0]
            neg_sorted_inds = np.argsort(cls[i, [neg_inds]])[0]
            k_cls_neg_hard = np.min([neg_inds.shape[0], this.hard_neg_samples])
            cls_neg_hard_indices = neg_inds[neg_sorted_inds][-k_cls_neg_hard:]
            k_cls_neg_rand = np.max([0, np.min([this.rand_neg_samples, neg_inds.shape[0] - k_cls_neg_hard])])
            if k_cls_neg_rand > 0:
                cls_neg_rand_indices = np.random.choice(neg_inds[neg_sorted_inds][:-k_cls_neg_hard],
                                                        size=k_cls_neg_rand, replace=False)
            else:
                cls_neg_rand_indices = np.array([], dtype=np.int)
            cls_inds = np.concatenate([pos_inds, cls_neg_hard_indices, cls_neg_rand_indices])
            cls_label_inds[i, cls_inds] = 1
            # for box
            pos_sorted_inds = np.argsort(box[i, [pos_inds]])[0]
            k_box_pos_hard = np.min([this.hard_reg_samples, pos_inds.shape[0]])
            box_pos_hard_indices = pos_inds[pos_sorted_inds][:k_box_pos_hard]
            k_box_pos_rand = np.max([0, np.min([this.rand_reg_samples, pos_inds.shape[0] - k_box_pos_hard]), 0])
            if k_box_pos_rand > 0:
                box_pos_rand_indices = np.random.choice(pos_inds[pos_sorted_inds][k_box_pos_hard:], size=k_box_pos_rand,
                                                        replace=False)
            else:
                box_pos_rand_indices = np.array([], dtype=np.int)
            box_inds = np.concatenate([box_pos_hard_indices, box_pos_rand_indices])
            box_label_inds[i, box_inds] = 1
            # for angle
            pos_sorted_inds = np.argsort(angle[i, [pos_inds]])[0]
            k_angle_pos_hard = np.min([this.hard_reg_samples, pos_inds.shape[0]])
            angle_pos_hard_indices = pos_inds[pos_sorted_inds][:k_angle_pos_hard]
            k_angle_pos_rand = np.max([0, np.min([this.rand_reg_samples, pos_inds.shape[0] - k_angle_pos_hard]), 0])
            if k_angle_pos_rand > 0:
                angle_pos_rand_indices = np.random.choice(pos_inds[pos_sorted_inds][k_angle_pos_hard:],
                                                          size=k_angle_pos_rand, replace=False)
            else:
                angle_pos_rand_indices = np.array([], dtype=np.int)
            angle_inds = np.concatenate([angle_pos_hard_indices, angle_pos_rand_indices])
            angle_label_inds[i, angle_inds] = 1
        return cls_label_inds.reshape(y_true_cls.shape), box_label_inds.reshape(
            y_true_cls.shape), angle_label_inds.reshape(y_true_cls.shape)

    def get_instance_balanced_weight(this,poly_mask):
        ib_weight = np.ones_like(poly_mask, dtype=np.float32)
        for n in range(poly_mask.shape[0]):
            total_area = np.where(poly_mask[n] != 0)[0].shape[0]
            poly_num = int(np.max(poly_mask[n]))
            poly_area_list = []
            for i in range(poly_num):
                poly_i = np.where(poly_mask[n] == i + 1, np.ones_like(poly_mask[n]), np.zeros_like(poly_mask[n]))
                area_i = np.sum(poly_i)
                if area_i == 0:
                    continue
                else:
                    poly_area_list.append((i + 1, area_i))
            # print(len(poly_area_list))
            if len(poly_area_list) == 0:
                continue
            B = float(total_area) / (len(poly_area_list)+this.EPS)
            for p, area_p in poly_area_list:
                ib_weight[n, :, :, :] = np.where(poly_mask[n] == p,
                                                 np.ones_like(poly_mask[n], dtype=np.float32) * B / (area_p+this.EPS),
                                                 ib_weight[n])
        return ib_weight

    def build_loss(this,y_true_cls, y_pred_cls,
                   y_true_geo, y_pred_geo,
                   training_mask, poly_mask,
                   with_ohem, with_instance_balance):
        '''
        define the loss used for training, contraning two part,
        the first part we use dice loss instead of weighted logloss,
        the second part is the iou loss defined in the paper
        :param y_true_cls: ground truth of text
        :param y_pred_cls: prediction os text
        :param y_true_geo: ground truth of geometry
        :param y_pred_geo: prediction of geometry
        :param training_mask: mask used in training, to ignore some text annotated by ###
        :return:
        '''

        classification_loss_n = this.cross_entropy_loss_with_mask(y_pred_cls, y_true_cls, training_mask)

        # d1 -> top, d2->right, d3->bottom, d4->left
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3,name="aho")
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=-1,name="bokeh");
        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
        w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
        h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        L_AABB = -tf.log((area_intersect + 1.0) / (area_union + 1.0))
        L_theta = 1 - tf.cos(theta_pred - theta_gt)
        box_loss_angle = L_theta * y_true_cls * training_mask
        box_loss_iou = L_AABB * y_true_cls * training_mask

        if with_ohem:
            cls_indices, box_indices, angle_indices = tf.py_func(this.sample_ohem,
                                                                 [y_true_cls, classification_loss_n, box_loss_iou,
                                                                  box_loss_angle], [tf.float32, tf.float32, tf.float32])
            cls_indices.set_shape([None, None, None, 1])
            box_indices.set_shape([None, None, None, 1])
            angle_indices.set_shape([None, None, None, 1])
            if with_instance_balance:
                ib_weights = tf.py_func(this.get_instance_balanced_weight, [poly_mask * training_mask], [tf.float32])[0]
                ib_weights.set_shape([None, None, None, 1])
                cls_indices *= ib_weights
                box_indices *= ib_weights
                angle_indices *= ib_weights
            classification_loss = tf.reduce_sum(classification_loss_n * cls_indices) / (tf.reduce_sum(cls_indices)+this.EPS)
            box_loss = tf.reduce_sum(box_loss_iou * box_indices) / (tf.reduce_sum(box_indices)+this.EPS)
            angle_loss = tf.reduce_sum(box_loss_angle * angle_indices) / (tf.reduce_sum(angle_indices)+this.EPS)
            tf.summary.scalar('classification_loss', classification_loss)
            tf.summary.scalar('geometry_AABB', box_loss)
            tf.summary.scalar('geometry_theta', angle_loss)
        else:
            if with_instance_balance:
                ib_weights = tf.py_func(this.get_instance_balanced_weight, [poly_mask], [tf.float32])[0]
                ib_weights.set_shape([None, None, None, 1])
                classification_loss = tf.reduce_mean(classification_loss_n * ib_weights)
                box_loss = tf.reduce_mean(box_loss_iou * ib_weights)
                angle_loss = tf.reduce_mean(box_loss_angle * ib_weights)
            else:
                classification_loss = tf.reduce_mean(classification_loss_n)
                box_loss = tf.reduce_mean(box_loss_iou)
                angle_loss = tf.reduce_mean(box_loss_angle)
            tf.summary.scalar('classification_loss', classification_loss)
            tf.summary.scalar('geometry_AABB', box_loss)
            tf.summary.scalar('geometry_theta', angle_loss)

        return [classification_loss , box_loss , angle_loss]



#

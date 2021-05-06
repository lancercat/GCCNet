import numpy as np;
class WELF_CONSTS:
    eps=1e-12;
class welf_basic_np_loss_helper:
    @staticmethod
    def get_instance_balanced_weight(poly_mask):
        """
        :param poly_mask:  The belonging of each point.
        :return:
        """
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
            B = float(total_area) / (len(poly_area_list) + WELF_CONSTS.eps)
            for p, area_p in poly_area_list:
                ib_weight[n, :, :, :] = np.where(poly_mask[n] == p,
                                                 np.ones_like(poly_mask[n], dtype=np.float32) * B / (
                                                         area_p + WELF_CONSTS.eps),
                                                 ib_weight[n])
        return ib_weight;

    @staticmethod
    def ohem_batch_over_group(valid_indicies, losses, hard_samples, rand_samples):
        # print(valid_indicies);
        # print(valid_indicies.shape);

        sorted_inds = valid_indicies[np.argsort(losses[valid_indicies])];
        k_hard = np.min([valid_indicies.shape[0], hard_samples]);
        hard_indices = sorted_inds[-k_hard:];
        k_rand = np.max([0, np.min([rand_samples, valid_indicies.shape[0] - k_hard])]);
        if k_rand > 0:
            rand_indices = np.random.choice(sorted_inds[:-k_hard],
                                            size=k_rand, replace=False)
        else:
            rand_indices = np.array([], dtype=np.int);
        indicies = np.concatenate([hard_indices, rand_indices])
        return indicies;

class welf_EAST_np_loss_helper:
    @staticmethod
    def sample_ohem_cls(label, neg_inds, pos_inds, cls_losses, hard_neg_samples, rand_neg_samples):
        ineg = welf_basic_np_loss_helper.ohem_batch_over_group(neg_inds, cls_losses, hard_neg_samples, rand_neg_samples);
        ipos = pos_inds;
        weight = np.zeros_like(label);
        all_cls_inds = np.concatenate([ineg, ipos]);
        weight[all_cls_inds] = 1;
        return weight;

    @staticmethod
    def sample_ohem_geo(label, pos_inds, losses, hard_samples, rand_samples):
        isel = welf_basic_np_loss_helper.ohem_batch_over_group(pos_inds, losses, hard_samples, rand_samples);
        weight = np.zeros_like(label);
        weight[isel] = 1;
        return weight;
    @staticmethod
    def sample_ohem_EAST_quad(y_true_cls, cls_loss, box_loss,
                         hard_neg_samples, rand_neg_samples,
                         hard_reg_samples, rand_reg_samples):

        label = y_true_cls.reshape((y_true_cls.shape[0], -1))
        sib=label.shape[1];
        hard_neg_samples = int(hard_neg_samples * sib);
        rand_neg_samples = int(rand_neg_samples * sib);
        hard_reg_samples = int(hard_reg_samples * sib);
        rand_reg_samples = int(rand_reg_samples * sib);
        #print(hard_neg_samples,hard_reg_samples);
        cls = cls_loss.reshape((cls_loss.shape[0], -1))
        box = box_loss.reshape((box_loss.shape[0], -1))
        cls_label_weights = np.zeros_like(label)
        box_label_weights = np.zeros_like(label)

        for i in range(label.shape[0]):
            blab = label[i];
            pos_inds = np.where(blab == 1)[0]
            neg_inds = np.where(blab == 0)[0]

            cls_label_weights[i] = welf_EAST_np_loss_helper.sample_ohem_cls(blab, neg_inds, pos_inds, cls[i], hard_neg_samples,
                                                                            rand_neg_samples)
            box_label_weights[i] = welf_EAST_np_loss_helper.sample_ohem_geo(blab, pos_inds, box[i], hard_reg_samples,
                                                                            rand_reg_samples);

        return cls_label_weights.reshape(y_true_cls.shape), box_label_weights.reshape(
             y_true_cls.shape);

    @staticmethod
    def sample_ohem_EAST(y_true_cls, cls_loss, box_loss, angle_loss,
                         hard_neg_samples, rand_neg_samples,
                         hard_reg_samples, rand_reg_samples):

        label = y_true_cls.reshape((y_true_cls.shape[0], -1))
        sib=label.shape[1];
        hard_neg_samples = int(hard_neg_samples * sib);
        rand_neg_samples = int(rand_neg_samples * sib);
        hard_reg_samples = int(hard_reg_samples * sib);
        rand_reg_samples = int(rand_reg_samples * sib);
        #print(hard_neg_samples,hard_reg_samples);
        cls = cls_loss.reshape((cls_loss.shape[0], -1))
        box = box_loss.reshape((box_loss.shape[0], -1))
        angle = angle_loss.reshape((angle_loss.shape[0], -1))
        cls_label_weights = np.zeros_like(label)
        box_label_weights = np.zeros_like(label)
        angle_label_weights = np.zeros_like(label)

        for i in range(label.shape[0]):
            blab = label[i];
            pos_inds = np.where(blab == 1)[0]
            neg_inds = np.where(blab == 0)[0]

            cls_label_weights[i] = welf_EAST_np_loss_helper.sample_ohem_cls(blab, neg_inds, pos_inds, cls[i], hard_neg_samples,
                                                                            rand_neg_samples)
            box_label_weights[i] = welf_EAST_np_loss_helper.sample_ohem_geo(blab, pos_inds, box[i], hard_reg_samples,
                                                                            rand_reg_samples);
            angle_label_weights[i] = welf_EAST_np_loss_helper.sample_ohem_geo(blab, pos_inds, angle[i], hard_reg_samples,
                                                                              rand_reg_samples);

        return cls_label_weights.reshape(y_true_cls.shape), box_label_weights.reshape(
            y_true_cls.shape), angle_label_weights.reshape(y_true_cls.shape)

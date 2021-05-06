import numpy as np;
from utils.libctpn.ctpn_anchor_target_layer import anchor_target_layer
import math;
class libctpn:
    # a box is an array of shape [4,2];
    # im_size_hw=[h,w]
    # Todo make it support "Just fit clipping" for skewed boxes
    @staticmethod
    def split_box_as_horizontal(box,im_size_hw,step = 16.0):
        pt_x = box[:,0].astype(int);
        pt_y = box[:, 1].astype(int);
        ind_x = np.argsort(pt_x, axis=0)
        pt_x = pt_x[ind_x]
        pt_y = pt_y[ind_x]

        if pt_y[0] < pt_y[1]:
            pt1 = (pt_x[0], pt_y[0])
            pt3 = (pt_x[1], pt_y[1])
        else:
            pt1 = (pt_x[1], pt_y[1])
            pt3 = (pt_x[0], pt_y[0])

        if pt_y[2] < pt_y[3]:
            pt2 = (pt_x[2], pt_y[2])
            pt4 = (pt_x[3], pt_y[3])
        else:
            pt2 = (pt_x[3], pt_y[3])
            pt4 = (pt_x[2], pt_y[2])

        xmin = int(min(pt1[0], pt2[0]))
        ymin = int(min(pt1[1], pt2[1]))
        xmax = int(max(pt2[0], pt4[0]))
        ymax = int(max(pt3[1], pt4[1]))

        if xmin < 0:
            xmin = 0
        if xmax > im_size_hw[1] - 1:
            xmax = im_size_hw[1] - 1
        if ymin < 0:
            ymin = 0
        if ymax > im_size_hw[0] - 1:
            ymax = im_size_hw[0] - 1

        width = xmax - xmin
        height = ymax - ymin

        # reimplement
        x_left = []
        x_right = []

        x_left.append(xmin)
        x_left_start = int(math.ceil(xmin / step) * step)
        if x_left_start == xmin:
            x_left_start = xmin + step
        for i in np.arange(x_left_start, xmax, step):
            x_left.append(i)
        x_left = np.array(x_left)

        x_right.append(x_left_start - 1)
        for i in range(1, len(x_left) - 1):
            x_right.append(x_left[i] + step-1);

        x_right.append(xmax)
        x_right = np.array(x_right)

        idx = np.where(x_left == x_right)

        x_left = np.delete(x_left, idx, axis=0)
        x_right = np.delete(x_right, idx, axis=0)
        nbs=[];
        for i in range(len(x_left)):
            nbs.append([
                [x_left[i],ymin],
                [x_right[i], ymin],
                [x_right[i], ymax],
                [x_left[i], ymax],
                ]);
        return nbs;
    @staticmethod
    def naive_b2atli(bs):
        is_hard=[];
        atlib=[];
        for b in bs:
            is_hard.append(0);
            atlib.append([b[0][0],b[0][1],b[2][0],b[2][1],1]);
        return np.array(atlib),np.array(is_hard);
    @staticmethod
    def split(boxes,im_size_hw):
        anbs=[]
        for b in boxes:
            anbs+=libctpn.split_box_as_horizontal(b,im_size_hw);
        return np.array(anbs);

    @staticmethod
    # todo add handling for dc regions.
    def generate_gt(im_size_hw,text_polys,text_tags,scale):
        sbs=libctpn.split(text_polys,im_size_hw);
        gt_boxes,gt_ishard=libctpn.naive_b2atli(sbs);
        dontcare_areas=np.zeros([0,4]);
        rpn_labels,rpn_reg,iw,ow=anchor_target_layer(gt_boxes,gt_ishard,dontcare_areas,im_size_hw[0],im_size_hw[1],scale,None);
        return rpn_labels,rpn_reg,iw,ow;


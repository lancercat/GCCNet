from utils.librnet.rnet_helpers import kot_rnet_helper;
from utils.geometry_utils import kot_min_area_quadrilateral,welf_crop_textline;

import numpy as np;
import cv2;
class kot_lib_rnet:
    @staticmethod
    def get_boxed(im,kps,inp_size):
        roi,_=kot_rnet_helper.get_roictx(kps,1);
        cc=kot_rnet_helper.clip_roi(im,roi);
        clip=cv2.resize(cc,inp_size);
        return clip;
    @classmethod
    def norm_imb(cls,im,boxes):
        imr = 1280 * 720 / im.shape[0] / im.shape[1];
        ims = [int(im.shape[1] * imr), int(im.shape[0] * imr)]
        p = ims[0] % 32;
        if (p):
            ims[0] = ims[0] + 32 - p;
        p = ims[1] % 32;
        if (p):
            ims[1] = ims[1] + 32 - p;
        xr = im.shape[1] / ims[0];
        yr = im.shape[0] / ims[1];

        ims = tuple(ims);
        im = cv2.resize(im, ims);
        for i in range(len(boxes)):
            boxes[i] = (boxes[i].reshape(-1,2) * np.array([xr, yr])).reshape(-1);
        return im,boxes;

    def generate_gt(this,package,im,inp_size,translator):
        ccs=[];
        tags=[];
        vals=[];
        boxes=[];
        for item in package.pred_items:
            cc,box=welf_crop_textline(im,item.kps);
            if(cc is None):
                continue;
            boxes.append(kot_min_area_quadrilateral(item.kps));

            if(len(inp_size)==2):
                cc=cv2.resize(cc, inp_size);
            else:
                ss=min(cc.shape[0:2]);
                rat=(inp_size[0]+1.)/ ss;
                s=[int(rat*cc.shape[1]),int(rat*cc.shape[0])];
                if(s[0]>160):
                    s[0]=160;
                if(s[1]>160):
                    s[1]=160;
                s=tuple(s);
                cc = cv2.resize(cc, s);
            im,boxes=this.norm_imb(im,boxes);
            ccs.append(cc);
            tag=item.gt_lab;
            val=1;
            if(tag<0):
                tag=0;
                val=0;
            tags.append(tag);
            vals.append(val);
        return [im,ccs,boxes,tags,vals];


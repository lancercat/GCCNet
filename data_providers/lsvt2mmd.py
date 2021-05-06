import json;
from data_providers.dataset import gt_desc_factory;
from utils.libpath import pathcfg;

import cv2;
import numpy as np;
import glob;
class convert:
    @classmethod
    def make_mask(cls,item):
        ret=[];
        im_i=cv2.imread(item.im_path);
        h,w,c=im_i.shape;
        raw=np.zeros([h,w],np.uint8);
        lens=[];
        for b in item.boxes:
            m=cv2.fillPoly(raw.copy(), [b.astype(int)],  1);
            # cv2.imshow("bax",m*250);
            # cv2.imshow("baa",im_i);
            # cv2.waitKey(0);
            ret.append(m);

            lens.append(len(b));
        return ret,item.boxes,lens;


    @classmethod
    def item_get_ann(cls,item):
        ad = {};
        bboxs = [];
        bboxs_ign = [];
        labels = [];
        labels_ign = [];
        for i in range(len(item.boxes)):
            box = cv2.boundingRect(item.boxes[i].astype(int));
            box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            if (item.det_dcs[i]):
                bboxs.append(box);
                labels.append(1);
            else:
                bboxs_ign.append(box);
                labels_ign.append(1);

        ad["bboxes"] = np.array(bboxs).tolist();
        ad["labels"] = np.array(labels).tolist();
        ad["bboxes_ignore"] = np.array(bboxs_ign).tolist()
        ad["labels_ignore"] = np.array(labels_ign).tolist();
        return ad;

    @classmethod
    def convert(cls,im,gt):
        item=gt_desc_factory.from_tencent(im,gt,None,None);
        im_i=cv2.imread(im);
        d={};
        d["filename"]=im;
        d["width"] = im_i.shape[1];
        d["height"]=im_i.shape[0];
        d["ann"]=cls.item_get_ann(item);
        return d;
    @classmethod
    def convert_ds(cls,ims,gts,tar_tr,tar_val):
        t=[];
        v=[];
        for i in range(len(ims)):
            if(i%5):
                t.append(cls.convert(ims[i],gts[i]));
            else:
                v.append(cls.convert(ims[i],gts[i]));

                print(i);
            if(i>130):
                break;

        json.dump(t, open(tar_tr, "w+"));

        json.dump(v, open(tar_val, "w+"));

    @classmethod
    def get_list(cls,img_f,gt_f):
        ims=glob.glob(img_f+"/*.jpg");
        ims.sort();
        gts=glob.glob(gt_f+ "/*.txt");
        gts.sort();
        return ims,gts;

if __name__ == '__main__':
    import os;
    from utils.libpath import pathcfg;
    gt_base = os.path.join(pathcfg.lsvt_train_dataset,"gt");
    im_base = os.path.join(pathcfg.lsvt_train_dataset,"img");

    a,b=convert.get_list(im_base,gt_base);
    convert.convert_ds(a,b,os.path.join(pathcfg.lsvt_train_dataset,"t.json"),os.path.join(pathcfg.lsvt_val_dataset,"v.json"));

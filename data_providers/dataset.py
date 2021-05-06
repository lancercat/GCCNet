from utils.geometry_utils import welf_crop_textline;
from utils.libpy import n_even_chunks;
from utils.libanno import lang_infer;
import  random;
import numpy as np;

import cv2;
import json;
import re;
from utils.geometry_utils import kot_min_area_quadrilateral;

class gt_desc:
    def __init__(this,im_path,
                 boxes,texts,langs,det_dcs,
                 ch_mask_path,ch_loc,ch_trans,ch_dcs,
                 has_gt,is_crop_word=False):
        this.im_path = im_path;
        this.ch_mask_path=ch_mask_path;
        this.ch_loc=ch_loc;
        this.ch_trans=ch_trans;
        this.boxes=boxes;
        this.texts=texts;
        this.langs=langs;
        this.det_dcs=det_dcs;
        this.ch_dcs=ch_dcs
        this.has_gt=has_gt;
        this.is_crop_word=is_crop_word;

    def aug_box(this,box):
        return box;
    def get_random_crop(this):
        if(this.is_crop_word):
            # the number here does not matter....
            cid=9;
        else:
            cid=random.choice(range(len(this.boxes)));
        return this.get_crop(cid),cid;
    def get_crop(this,cid):
        if(this.is_crop_word):
            return cv2.imread(this.im_path), this.texts[0];
        im=cv2.imread(this.im_path);
        box=this.boxes[cid].copy();
        box=this.aug_box(box);
        return welf_crop_textline(im,np.array(box))[0],this.texts[cid];

    def get_crop_cnt(this):
        if(this.boxes is None):
            return 0;
        else:
            return len(this.boxes);

    def extract_crop_indexs(this):
        return range(len(this.boxes));


class gt_desc_factory:
    @staticmethod
    def is_dc(str,regex=r"^#+$"):
        return re.match(regex,str) is not None;
    @staticmethod
    def from_icdar(im_path, gt_path, mask_paths, char_gt_paths):
        f = open(gt_path, "r");
        boxes = [];
        texts = [];
        langs = [];
        dec_dcs = [];

        for line in f:
            if(len(line)<5):
                continue
            fields = line.split(",", 9);
            fields[-1]=fields[-1].replace("\n","");
            fields[-1]=fields[-1].replace("\r","");
            box = [];
            for i in range(4):
                box.append([float(fields[i * 2 ].replace("\ufeff","")), float(fields[i * 2 + 1].replace("\ufeff",""))]);
            boxes.append(box);
            texts.append(fields[-1]);
            if(gt_desc_factory.is_dc(fields[-1])):
                dec_dcs.append(1);
                langs.append(lang_infer.LANG_DC);
            else:
                langs.append(lang_infer.LANG_EN);
                dec_dcs.append(0);
        f.close();
        return gt_desc(im_path,
                       boxes,texts,langs,dec_dcs,
                       mask_paths,None,None,None,
                       True,False);


        pass;
    @staticmethod
    def from_tencent(im_path, gt_path, mask_paths, char_gt_paths):
        boxes=[];
        texts=[];
        dec_dcs=[];
        langs=[];

        lines= open(gt_path, "r");
        for line in lines:
            ffs = line.split("\"");
            fs = ffs[0].split(",")
            cords = [];
            for i in fs:
                if (len(i)):
                    cords.append(float(i));
            if (len(cords) < 6):
                continue;
            kps = np.array(cords).reshape(-1, 2);
            box = kot_min_area_quadrilateral(kps);
            if(len(ffs)>1):
                txt = ffs[1];
            else:
                txt="The result only goddesses know";
            boxes.append(box);
            texts.append(txt);
            try:
                if (gt_desc_factory.is_dc(txt[-1])):
                    dec_dcs.append(1);
                    langs.append(lang_infer.LANG_DC);
                else:
                    langs.append(lang_infer.LANG_EN);
                    dec_dcs.append(0);
            except:
                dec_dcs.append(1);
                langs.append(lang_infer.LANG_DC);
                pass;

        lines.close();
        return gt_desc(im_path,
                       boxes, texts, langs, dec_dcs,
                       mask_paths, None, None, None,
                       True, False);
    @staticmethod
    def from_rects(im_path, gt_path, mask_paths, char_gt_paths):
        f = open(gt_path, "r");
        d=json.load(f);
        ld=d["lines"];
        cd=d["chars"];
        boxes=[];
        texts=[];
        dec_dcs=[];
        langs=[];

        chs=[];
        ch_dcs=[];
        ch_trans=[];
        ch_langs=[];

        for line in ld:
            box=kot_min_area_quadrilateral(np.array(line["points"]).astype(float).reshape(-1, 2));
            boxes.append(box.tolist());
            texts.append(line["transcription"]);
            if (line["ignore"]):
                dec_dcs.append(1);
                langs.append(lang_infer.LANG_DC);
            else:
                langs.append(lang_infer.infer_lang(line["transcription"].replace('#', '')));
                dec_dcs.append(0);
        for c in cd:
            box = kot_min_area_quadrilateral(np.array(c["points"]).astype(float).reshape(-1, 2));
            chs.append(box.tolist());
            ch_trans.append(c["transcription"]);
            if (c["ignore"]):
                ch_dcs.append(1);
                ch_langs.append(lang_infer.LANG_DC);
            else:
                ch_langs.append(lang_infer.infer_lang(c["transcription"].replace('#', '')));
                ch_dcs.append(0);

        f.close();
        return gt_desc(im_path,
                       boxes, texts, langs, dec_dcs,
                       mask_paths, chs,ch_trans,ch_dcs,
                       True, False);

    @staticmethod
    def from_crawler(im_path):
        return gt_desc(im_path,
                       None,None,None,None,
                       None,None,None,None,
                       False,False);

    @staticmethod
    def from_tencent_crop_word(im_path, gt_path, mask_paths, char_gt_paths):
        im=cv2.imread(im_path);
        h,w=im.shape[0],im.shape[1];
        boxes=[[0,0],[w,0],[w,h],[0,h]];
        [im,gt]=im_path.split("->-",1);
        texts=[gt];
        langs=[lang_infer.infer_lang(gt)];
        return gt_desc(im,
                       boxes, texts, langs, None,
                       None,None, None,None,
                       True,True);

    @staticmethod
    def from_rctw(im_path,gt_path,mask_paths,char_gt_paths):
        f=open(gt_path,"r", encoding='utf-8-sig');
        boxes=[];
        texts=[];
        langs=[];
        dec_dcs=[];

        for line in f:
            fields=line.split(",",9);
            fields[-1] = fields[-1][1:];
            fields[-1]=fields[-1].replace("\n","");
            fields[-1]=fields[-1].replace("\r","");
            fields[-1]=fields[-1].strip("\"");
            box = [];
            for i in range(4):
                box.append([int(fields[i * 2 ]), int(fields[i * 2 + 1])]);
            boxes.append(box);
            texts.append(fields[-1]);
            if(gt_desc_factory.is_dc(fields[-1])):
                langs.append(lang_infer.LANG_DC);
                dec_dcs.append(1);
            else:
                langs.append(lang_infer.infer_lang(fields[-1].replace('#','')));
                dec_dcs.append(0);
        f.close();
        return gt_desc(im_path,
                       boxes,texts,langs,dec_dcs,
                       mask_paths,None,None,None,
                       True,False);


    @staticmethod
    def from_mlt(im_path, gt_path, mask_paths, char_gt_paths):
        f = open(gt_path, "r");
        boxes = [];
        texts = [];
        langs = [];
        dec_dcs = [];
        try:
            for line in f:
                fields = line.split(",", 9);
                fields[-1]=fields[-1].replace("\n","");
                fields[-1]=fields[-1].replace("\r","");
                box = [];
                for i in range(4):
                    box.append([int(fields[i * 2 ]), int(fields[i * 2 + 1])]);
                boxes.append(box);
                texts.append(fields[-1]);
                if(gt_desc_factory.is_dc(fields[-1])):
                    dec_dcs.append(1);
                    langs.append(lang_infer.LANG_DC);
                else:
                    langs.append(lang_infer.get(fields[-1]));
                    dec_dcs.append(0);
        except:
            print(gt_path);
            Nah.man.thisWontDo;
        f.close();
        return gt_desc(im_path,
                       boxes,texts,langs,dec_dcs,
                       mask_paths,None,None,None,
                       True,False);

    @staticmethod
    def build_all(im_paths,gt_paths,mask_paths,char_gt_paths,type):
        factory_func=None
        if(type=="rctw"):
            factory_func=gt_desc_factory.from_rctw;
        elif (type=="mlt"):
            factory_func=gt_desc_factory.from_mlt;
        elif (type=="icdar"):
            factory_func=gt_desc_factory.from_icdar;
        elif(type=="taocr"):
            factory_func=gt_desc_factory.from_tencent_crop_word;
        elif(type=="rects"):
            factory_func = gt_desc_factory.from_rects;
        elif (type == "tencent"):
            factory_func = gt_desc_factory.from_tencent;

        else:
            print(type);
            exit(9);
        dataset=[];
        tot=len(im_paths);
        for i in range(tot):
            if(i%100==0):
                print(i,"in",tot);
            # if(i>100):
            #     break;
            dataset.append(factory_func(im_paths[i],gt_paths[i],mask_paths[i],char_gt_paths[i]));
        print("done_loading");
        return dataset;



    @staticmethod
    def build_all_with_index(index_path,source_type):
        f=open(index_path,"r");
        damn=[];
        for line in f:
            damn.append(line.strip());
        f.close();
        imgs,masks,ch_gts,gts=n_even_chunks(damn,4);
        return gt_desc_factory.build_all(imgs,gts,masks,ch_gts,source_type);


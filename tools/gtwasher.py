import glob;
import cv2;
import os;
from utils.libanno import lang_infer;
import utils.libnorm as libnorm;
from data_providers.dataset import gt_desc_factory;
import numpy as np;


class neko_washer:
    def __init__(this,configs):
        this.entries = [];
        for i in configs:
            this.entries+=gt_desc_factory.build_all_with_index(i["index"],i["type"]);

    def save(this,crop_dir,crop_index_fp,fn):
        cid=0;
        for eid in range(len(this.entries)):
            # gt=gt_desc;
            gt=this.entries[eid];
            for i in range(len(gt.texts)):
                if(fn(gt,i)):
                    img,txt=gt.get_crop(i);
                    cv2.imwrite(os.path.join(crop_dir,str(cid)+".jpg"),img);
                    crop_index_fp.write(str(cid)+".jpg"+"->-"+txt+"\n");
                    cid+=1;
            if(eid%100==0):
                print(eid/len(this.entries));
                crop_index_fp.flush();



configs=[{"index":"/home/lasercat/pubdata/rctw/a.index","type":"rctw"},{"index":"/home/lasercat/pubdata/mlt/a.index","type":"mlt"}];
out_dir="/home/lasercat/shareddata/pubdata/crop/";
out_fp=open("/home/lasercat/shareddata/pubdata/crop/index.txt","w+");

c=neko_washer(configs);
c.save(out_dir,out_fp,neko_washer.horiziontal_h_mix);
out_fp.close();

print(c.entries);

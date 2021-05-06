import json;
import glob;
import os;
from utils.libpath import pathcfg;

from data_providers.dataset import gt_desc_factory
def cvt_file(src,dst):
    f=gt_desc_factory();
    g=f.from_rects("dummy",src,"dummy","dummy");
    length=len(g.det_dcs);
    lines=[];
    for i in range(len(g.boxes)):
        res="";
        box=g.boxes[i];
        trans=g.texts[i];
        dc=g.det_dcs[i];
        for i in range(4):
            for j in range(2):
                res+=str(int(box[i][j]));
                res+=",";
        if(dc):
            res+="###";
        else:
            res+=trans.strip("\n");
        res+="\n";
        lines.append(res);
    with open(dst,"w") as fp:
        fp.writelines(lines);
        fp.close();

    return;
srcs=glob.glob("/home/lasercat/pubdata/datasets/cat/rects_val/json_gt/*.json");
ddir="/home/lasercat/pubdata/datasets/cat/rects_val/gt/";

for i in srcs:
    cvt_file(i,os.path.join(ddir,os.path.basename(i).replace("json","txt")));



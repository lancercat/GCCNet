import os;

import numpy as np;
import glob;
class convert:
    @staticmethod
    def welf_rrect_to_4pts(center, size, angle):
        _angle = angle
        b = np.cos(_angle) * 0.5
        a = np.sin(_angle) * 0.5

        pt = [0 for _ in range(8)]
        pt[0] = center[0] - a * size[1] - b * size[0]
        pt[1] = center[1] + b * size[1] - a * size[0]
        pt[2] = center[0] + a * size[1] - b * size[0]
        pt[3] = center[1] - b * size[1] - a * size[0]
        pt[4] = 2 * center[0] - pt[0]
        pt[5] = 2 * center[1] - pt[1]
        pt[6] = 2 * center[0] - pt[2]
        pt[7] = 2 * center[1] - pt[3]
        return np.array(pt, dtype=np.float).reshape(4, 2)

    @classmethod
    def convert(cls,gt,dst):
        ls=[i for i in open(gt,"r")];
        nls=[];
        for l in ls:
            fs=[int (i) for i in l.split(" ")[:7]];
            txt="text\n";
            if fs[1]:
                txt="###\n";
            box=cls.welf_rrect_to_4pts([fs[2],fs[3]],[fs[4],fs[5]],fs[6]).reshape(-1);
            nl="";
            for i in range(8):
                nl+=str(box[i]);
                nl+=",";
            nl+=txt;
            nls.append(nl);
        with open(dst,"w+") as f:
            f.writelines(nls);

    @classmethod
    def convert_ds(cls,gts,tar_tr):
        gts=glob.glob(gts+ "/*.txt");
        for gt in gts:
            key=os.path.basename(gt);
            cls.convert(gt,os.path.join(tar_tr,key));




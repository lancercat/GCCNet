
import utils.libnorm as libnorm;
import cv2;
import numpy as np;

class lang_infer:
    LANG_EN=0;
    LANG_CN=1;
    LANG_MIX=2;
    LANG_UK=3;
    LANG_DC=4;

    LANG_DICT={"Latin":LANG_EN,"Mixed":LANG_MIX,"Chinese":LANG_CN,"DC":LANG_DC};


    @staticmethod
    def infer_lang(str):
        eng = False
        Chinese = False
        for ch in str:
            if u'\u4e00' <= ch <= u'\u9fff':
                Chinese = True
            elif ch.isalnum():
                eng = True
        if Chinese and eng:
            return  lang_infer.LANG_MIX # mix
        elif Chinese:
            return lang_infer.LANG_CN  # Chinese
        elif eng:
            return lang_infer.LANG_EN  # English
        else:
            return lang_infer.LANG_UK  # others

    @staticmethod
    def get(typestr):
        if(typestr in lang_infer.LANG_DICT):
            return lang_infer.LANG_DICT[typestr];
        else:
            return lang_infer.LANG_UK;

class filters:
    @staticmethod
    def horiziontal_h_mix(desc,i):
        # desc=gt_desc;
        if(desc.langs[i] is lang_infer.LANG_UK or
                desc.langs[i] is lang_infer.LANG_DC ):
            return False;
        if(desc.boxes is None):
            return True;
        sz=libnorm.norm_sz(desc.boxes[i]);
        x,y,w,h=cv2.boundingRect(np.array(desc.boxes[i]));
        sz=[w,h];
        if(sz[0]*1.2<sz[1]):
            return False;
        return True;

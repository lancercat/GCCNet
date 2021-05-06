import numpy as np;
import cv2;
import math;
def distance(src,dst):
    td=0;
    for i in range(4):
        dv=(src[i]-dst[i]);
        d=dv[0]*dv[0]+dv[1]*dv[1];
        d*=d;
        d=math.sqrt(d);
        td+=d;
    return td;

def mapping(src,dst):
    best=0xca71*0xca39;
    best_mapping=9;
    for i in range(4):
        src=np.roll(src, 1, axis=0);
        d=distance(src,dst);
        if(d<best):
            best=d;
            best_mapping=src.copy();
    return best_mapping;

def norm_sz(cords):
    src_cords = np.float32(cords);
    rect = cv2.minAreaRect(src_cords);
    sz = (max(rect[1][0], rect[1][1]), min(rect[1][0], rect[1][1]));
    return sz;

def norm(src,cords,size):
    src_cords = np.float32(cords);
    if(size is None):
        sz=norm_sz(cords);
    else:
        sz=size;
    dst_cords=np.float32([[0,0],[sz[0],0],[sz[0],sz[1]],[0,sz[1]]]);
    src_cords=mapping(src_cords,dst_cords);

    M=cv2.getPerspectiveTransform(src_cords,dst_cords);
    dst=cv2.warpPerspective(src,M,(int(sz[0]),int(sz[1])));
    return dst;

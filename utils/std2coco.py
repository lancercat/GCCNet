import glob;
import os;

def convert_line(src):
    raws=src.strip().split(",");
    if(len(raws)<5):
        return None;
    xs=[];
    ys=[];
    for i in range(4):
        xs.append(float(raws[i*2]));
        ys.append(float(raws[i*2+1]));
    xmax=max(xs);
    ymax = max(ys);
    xmin = min(xs);
    ymin = min(ys);
    xmin=max(xmin,0);
    ymin = max(ymin,0);

    ret=str(int(xmin))+","+str(int(ymin))+","+str(int(xmax))+","+str(int(ymax))+","+"1\n";
    return ret;

def convert_file(src,dst):
    sfp=open(src,"r");
    lines=[];
    for i in sfp:
        ret=convert_line(i);
        if ret is not None:
            lines.append(ret);
    if len(lines):
        dfp=open(dst,"w+");
        dfp.writelines(lines);
        dfp.close();
    sfp.close();
def convert_dir(src,dst):
    try:
        os.rmdir(dst);
    except :
        pass
    try:
        os.mkdir(dst);
    except:
        pass;

    srcs=glob.glob(src+"*.txt");
    for s in srcs:
        bn=os.path.basename(s);
        d=os.path.join(dst,bn);
        convert_file(s,d);

convert_dir("/home/lasercat/cat/project_v_data/coco/b_al1259974submit/","/home/lasercat/cat/project_v_data/coco/b_al1259974submitcoco/")


import tensorboardX;
from  utils.libpath import pathcfg;
import os;
import shutil
import numpy as np;
# this is going to replace the old logger
class kot_logger:
    writter=None;
    @classmethod
    def init(cls,**args):
        path=args.get("dir",pathcfg.logs_root);
        path=os.path.join(path,"logs");
        shutil.rmtree(path, True);
        try:
            os.makedirs(path);
        except:
            pass;
        cls.writter=tensorboardX.SummaryWriter(path);

    @classmethod
    def nammedWindow(cls,*args, **kwargs):
        pass;
    @classmethod
    def imshow(cls,winname,im):
        if(im.shape[-1]==1):
            im=np.concatenate([im,im,im],-1);
        cls.writter.add_images(winname,np.expand_dims(im,0),dataformats="NHWC");
    @classmethod
    def waitKey(self,_):
        pass;

    @classmethod
    def log_image_array(cls,tag,iter,ims):
        cls.writter.add_images(tag,ims,iter,dataformats="NHWC");

    @classmethod
    def log_image(cls,tag,iter,im):
        cls.writter.add_images(tag,np.expand_dims(im,0),iter,dataformats="NHWC");

    @classmethod
    def log_scaler(cls,tag,iter,number):
        cls.writter.add_scalar(tag,number,iter);


    @classmethod
    def log_text(cls,tag,iter,text):
        cls.writter.add_text(tag,text,iter);

    @classmethod
    def log_image_array_h(cls, tag, iter, ims,mean,var):
        ims*=var;
        ims+=mean;
        ims=np.clip(ims,0,255);
        ims=ims.astype(np.uint8);
        ims = ims.astype(np.float32);

        ims/=255;
        cls.writter.add_images(tag, ims, iter, dataformats="NHWC");
import cv2;

class kot_fots_augumentor:
    @staticmethod
    def get_rnet_augumented(raw_entry,mean,var):

        im=cv2.imread(raw_entry.im_path).astype(float);
        im-=mean;
        im/=var;

        return im,raw_entry;


#       this.im_path = im_path;
#         this.ch_mask_path=ch_mask_path;
#         this.ch_loc=ch_loc;
#         this.ch_trans=ch_trans;
#         this.boxes=boxes;
#         this.texts=texts;
#         this.langs=langs;
#         this.det_dcs=det_dcs;
#         this.ch_dcs=ch_dcs
#         this.has_gt=has_gt;
#         this.is_crop_word=is_crop_word;
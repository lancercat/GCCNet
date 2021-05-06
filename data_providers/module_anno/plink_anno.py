from utils.libplink.gt_utils import cal_gt_for_single_image;


class libplink:
    def __init__(this,config):

        this.neibour_type=config.get(str,"neibour_type");
        this.bbox_border_width=config.get(float,"bbox_border_width");
        this.pixel_cls_border_weight_lambda=config.get(float,"pixel_cls_border_weight_lambda");
        this.method=config.get(str,"method");

    # todo add handling for dc regions.
    def generate_gt(this,im_size_hw,text_polys,text_tags,scale):
        # print("generating")
        return cal_gt_for_single_image(im_size_hw,text_polys,text_tags,scale,this.method,this.neibour_type,this.bbox_border_width,this.pixel_cls_border_weight_lambda);

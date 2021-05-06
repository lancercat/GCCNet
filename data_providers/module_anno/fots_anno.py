from utils.geometry_utils import kot_min_area_quadrilateral;
from data_providers.module_anno.ocr_anno import kot_translator;
from data_providers.dataset import gt_desc
class kot_lib_detrec:
    def __init__(this,config):
        dict_loc=config.get(str,"dict_path");
        this.ocr_core=kot_translator(dict_loc);
        pass;

    def generate_gt(this, raw_entry:gt_desc):
        tags = [];
        vals = [];
        boxes = [];
        txts=[];
        txt_masks=[];
        for i in range(len(raw_entry.boxes)):
            tags.append(0==raw_entry.det_dcs[i]);
            vals.append(0==raw_entry.det_dcs[i]);
            boxes.append(raw_entry.boxes[i]);
            txt,mask=this.ocr_core.translate(raw_entry.texts[i]);
            txts.append(txt);
            txt_masks.append(mask);

        return [boxes,txts,txt_masks, tags, vals];


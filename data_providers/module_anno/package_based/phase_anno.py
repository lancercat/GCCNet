from utils.geometry_utils import kot_min_area_quadrilateral;
from data_providers.module_anno.ocr_anno import kot_translator;

class kot_lib_phase_vanilla:
    BOXFN=kot_min_area_quadrilateral;
    def get_basic_targets(this,item):
        box=kot_min_area_quadrilateral(item.kps);
        val = item.lab > 0;
        lab = item.lab * val;
        return box,val,lab;
    def generate_gt(this,package):
        tags=[];
        vals=[];
        boxes=[];
        pred_boxes=[];
        for item in package.gt_items:
            box,val,lab=this.get_basic_targets(item);
            boxes.append(box);
            vals.append(val);
            tags.append(lab);
        for item in package.pred_items:
            pred_boxes.append(kot_min_area_quadrilateral(item.kps));
        return [boxes,tags,vals,pred_boxes];
#boxes,txts,txt_masks, tags, vals, pred_boxes

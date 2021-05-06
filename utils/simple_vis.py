from utils.det_box_cvtr import sort_poly;
import numpy as np;
import cv2;

class det_vis:
    @staticmethod
    def boxes_on_im(oim, boxes, color,filter_small=True):
        im = oim.copy();
        boxes = boxes[:, :8].reshape((-1, 4, 2))
        for box in boxes:
            # to avoid submitting errors
            box = sort_poly(box.astype(np.float32))
            if filter_small and( np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5):
                continue
            cv2.polylines(im, [box.astype(np.int32).reshape((-1, 1, 2))], True, color=color,
                          thickness=1)
        return im;

    @staticmethod
    def polys_on_im(oim, boxes, color):
        im = oim.copy();
        for box in boxes:
            # to avoid submitting errors
            cv2.polylines(im, [box.astype(np.int32).reshape((-1, 1, 2))], True, color=color,
                          thickness=3)
        return im;
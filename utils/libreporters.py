import os;
import numpy as np;
from utils.det_box_cvtr import sort_poly;
import cv2;


def i15_res_reporter(output_dir, im_fn, boxes,confs):
    res_file = os.path.join(output_dir,
                            'res_{}.txt'.format(os.path.basename(im_fn).split('.')[0]))
    with open(res_file, 'w') as f:
        for i in range(len(boxes)):
            # to avoid submitting errors
            box=boxes[i];
            conf=confs[i];
            box = sort_poly(box.astype(np.int32))
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            f.write('{},{},{},{},{},{},{},{},{}\r\n'.format(
                box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0],
                box[3, 1],conf
            ))


def coco_res_reporter(output_dir, im_fn, boxes,confs):
    res_file = os.path.join(output_dir,
                            'res_{}.txt'.format(os.path.basename(im_fn).split('.')[0]))
    with open(res_file, 'w') as f:
        for i in range(len(boxes)):
            # to avoid submitting errors
            box=boxes[i];
            conf=confs[i];
            box = sort_poly(box.astype(np.int32))

            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            x, y, w, h = cv2.boundingRect(box);

            f.write('{},{},{},{},{}\r\n'.format(x, y, x + w, y + h, conf));

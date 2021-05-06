import numpy as np;
from utils.det_box_cvtr import restore_rectangle_rbox as restore_rectangle;
from utils.lanms import merge_quadrangle_n9 as nms;
#from utils.welf_nms import standard_nms as nms;
import cv2;

class cat_east_paser:
    def __init__(this,score_map_thresh=0.9):
        this.score_map_thresh=score_map_thresh;

    def detect_2_boxes(this,preds,im_info,**kwargs):

        '''
        restore text boxes from score map and geo map
        :param score_map:
        :param geo_map:
        :param score_map_thresh: threshhold for score map
        :param box_thresh: threshhold for boxes
        :param nms_thres: threshold for nms
        :return:
        '''
        score_map=preds[0];
        geo_map=preds[1];
        if len(score_map.shape) == 4:
            score_map = score_map[0, :, :, 0]
            geo_map = geo_map[0, :, :, ]
        # filter the score map
        xy_text = np.argwhere(score_map > this.score_map_thresh)
        # sort the text boxes via the y axis
        xy_text = xy_text[np.argsort(xy_text[:, 0])]
        # restore
        text_box_restored = restore_rectangle(xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2
        print('{} text boxes before nms'.format(text_box_restored.shape[0]))
        boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
        boxes[:, :8] = text_box_restored.reshape((-1, 8))
        boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
        return boxes;

    def detect_2_boxes_rew(this, preds, im_info, **kwargs):

        '''
        restore text boxes from score map and geo map
        :param score_map:
        :param geo_map:
        :param score_map_thresh: threshhold for score map
        :param box_thresh: threshhold for boxes
        :param nms_thres: threshold for nms
        :return:
        '''
        score_map = preds[0];
        geo_map = preds[1];
        if len(score_map.shape) == 4:
            score_map = score_map[0, :, :, 0]
            geo_map = geo_map[0, :, :, ]
        # filter the score map
        xy_text = np.argwhere(score_map > this.score_map_thresh)
        # sort the text boxes via the y axis
        xy_text = xy_text[np.argsort(xy_text[:, 0])]
        # restore
        geo_max=np.max(geo_map[:,:,0:4],axis=-1);
        geo_min = np.max(geo_map[:, :, 0:4], axis=-1);
        # xy_big=np.argwhere(geo_max > 64);
        # xy_too_big=np.argwhere(geo_max > 128);
        xy_way_too_big = np.argwhere(geo_max > 384);
        xy_too_small = np.argwhere(geo_min < 7);
        # score_map[xy_big[:, 0], xy_big[:, 1]] *= 0.95;
        # score_map[xy_too_big[:, 0], xy_too_big[:, 1]] *= 0.97;
        score_map[xy_way_too_big[:, 0], xy_way_too_big[:, 1]] *=0.8;
        score_map[xy_too_small[:, 0], xy_too_small[:, 1]] *= 0.8 ;

        nsm=score_map.copy();
        # nsm[xy_big[:, 0],xy_big[:, 1]]*=0.8;
        # nsm[xy_too_big[:, 0],xy_too_big[:, 1]]*=0.8;
        # nsm[xy_way_too_big[:, 0],xy_way_too_big[:, 1]] *= 0.8;
        # nsm[xy_too_small[:,0],xy_too_small[:,1]] *= 0.8;
        text_box_restored = restore_rectangle(xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2
        print('{} text boxes before nms'.format(text_box_restored.shape[0]))
        boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
        boxes[:, :8] = text_box_restored.reshape((-1, 8))
        boxes[:, 8] = nsm[xy_text[:, 0], xy_text[:, 1]]
        return boxes;

    def filter_boxes_nr(this, boxes, with_nms=1,nms_thres=0.2):
        if (with_nms):
            boxes = nms(boxes.astype('float32'), nms_thres)
        if boxes.shape[0] == 0:
            return None
        return boxes;

    def filter_boxes(this,boxes, score_map, with_nms=1, box_thresh=0.0, nms_thres=0.2, fmp_scale=4):
        if (with_nms):
            boxes = nms(boxes.astype('float32'), nms_thres)

        if boxes.shape[0] == 0:
            return None
        score=score_map[0,:,:,0];
            # here we filter some low score boxes by the average score map, this is different from the orginal paper
        for i, box in enumerate(boxes):
            mask = np.zeros_like(score, dtype=np.uint8)
            mask=cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // fmp_scale, 1)
            boxes[i, 8] = cv2.mean(score, mask)[0]
        boxes = boxes[boxes[:, 8] > box_thresh]
        return boxes;

        #
        # fboxes=boxes[:,:8].reshape((-1, 4, 2)).astype(np.int32).copy();
        # print(score_map.shape[0]);
        # print(score_map.shape[1]);
        #
        # fboxes[:, :, 0] = np.clip(fboxes[:, :, 0], 0, score_map.shape[0]//fmp_scale)
        # fboxes[:, :, 1] = np.clip(fboxes[:, :, 1], 0, score_map.shape[1]//fmp_scale)
        # for i, fbox in enumerate(fboxes):
        #     mask = np.zeros_like(score_map, dtype=np.uint8)
        #     score_map=cv2.fillPoly(mask, fbox[:8].reshape((-1, 4, 2)).astype(np.int32) // fmp_scale, 1)
        #     boxes[i, 8] = cv2.mean(score_map, mask)[0]
        # boxes = boxes[boxes[:, 8] > box_thresh]
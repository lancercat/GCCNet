import numpy as np;
import cv2;
from Polygon import Polygon;
from utils.geometry_utils import shrink_poly;
from utils.geometry_utils import fit_line;
from utils.geometry_utils import neko_points_to_line,neko_points_to_each_line;
from utils.geometry_utils import neko_point_dist_to_line as point_dist_to_line ;

from utils.geometry_utils import line_cross_point;
from utils.geometry_utils import sort_rectangle;
from utils.det_box_cvtr import rectangle_from_parallelogram;


class kot_libeast:
    def __init__(this,config):
        this.margin_dont_care = config.get(int, "margin_dc");
        this.min_text_size= config.get(int, "min_text_size");
    def poly_2_para(this,poly):
        # if geometry == 'RBOX':
        # 对任意两个顶点的组合生成一个平行四边形
        fitted_parallelograms = []
        for i in range(4):
            p0 = poly[i]
            p1 = poly[(i + 1) % 4]
            p2 = poly[(i + 2) % 4]
            p3 = poly[(i + 3) % 4]
            edge = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            backward_edge = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            forward_edge = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            if point_dist_to_line(p0, p1, p2) > point_dist_to_line(p0, p1, p3):
                # 平行线经过p2
                if edge[1] == 0:
                    edge_opposite = [1, 0, -p2[0]]
                else:
                    edge_opposite = [edge[0], -1, p2[1] - edge[0] * p2[0]]
            else:
                # 经过p3
                if edge[1] == 0:
                    edge_opposite = [1, 0, -p3[0]]
                else:
                    edge_opposite = [edge[0], -1, p3[1] - edge[0] * p3[0]]
            # move forward edge
            new_p0 = p0
            new_p1 = p1
            new_p2 = p2
            new_p3 = p3
            new_p2 = line_cross_point(forward_edge, edge_opposite)
            if (new_p2 is None):
                print("emmmm");
                return None;
            if point_dist_to_line(p1, new_p2, p0) > point_dist_to_line(p1, new_p2, p3):
                # across p0
                if forward_edge[1] == 0:
                    forward_opposite = [1, 0, -p0[0]]
                else:
                    forward_opposite = [forward_edge[0], -1, p0[1] - forward_edge[0] * p0[0]]
            else:
                # across p3
                if forward_edge[1] == 0:
                    forward_opposite = [1, 0, -p3[0]]
                else:
                    forward_opposite = [forward_edge[0], -1, p3[1] - forward_edge[0] * p3[0]]
            new_p0 = line_cross_point(forward_opposite, edge)
            new_p3 = line_cross_point(forward_opposite, edge_opposite)
            fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
            # or move backward edge
            new_p0 = p0
            new_p1 = p1
            new_p2 = p2
            new_p3 = p3
            new_p3 = line_cross_point(backward_edge, edge_opposite)
            if (new_p0 is None or new_p1 is None or new_p2 is None or new_p3 is None):
                print("emmmm");
                return None;
            if point_dist_to_line(p0, p3, p1) > point_dist_to_line(p0, p3, p2):
                # across p1
                if backward_edge[1] == 0:
                    backward_opposite = [1, 0, -p1[0]]
                else:
                    backward_opposite = [backward_edge[0], -1, p1[1] - backward_edge[0] * p1[0]]
            else:
                # across p2
                if backward_edge[1] == 0:
                    backward_opposite = [1, 0, -p2[0]]
                else:
                    backward_opposite = [backward_edge[0], -1, p2[1] - backward_edge[0] * p2[0]]
            new_p1 = line_cross_point(backward_opposite, edge)
            new_p2 = line_cross_point(backward_opposite, edge_opposite)
            if (new_p1 is None or new_p2 is None):
                print("emmmm");
                return None;
            fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
        areas = [Polygon(t).area() for t in fitted_parallelograms]
        parallelogram = np.array(fitted_parallelograms[np.argmin(areas)][:-1], dtype=np.float32);
        return parallelogram;

    # **kwargs are for mutable args among images. like blur or not, ugly or not, etc
    def generate_gt(this,im_size_hw, text_polys, text_tags,scale=4,**kwargs ):
        h, w = im_size_hw
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        score_map = np.zeros((h, w), dtype=np.uint8)
        geo_map = np.zeros((h, w, 5), dtype=np.float32)
        # mask used during traning, to ignore some hard areas
        training_mask = np.ones((h, w), dtype=np.uint8)
        for poly_idx, poly_tag in enumerate(zip(text_polys, text_tags)):
            poly = poly_tag[0]
            tag = poly_tag[1]

            r = [None, None, None, None]
            for i in range(4):
                r[i] = min(np.linalg.norm(poly[i] - poly[(i + 1) % 4]),
                           np.linalg.norm(poly[i] - poly[(i - 1) % 4]))
            # score map
            shrinked_poly = shrink_poly(poly.copy(), r).astype(np.int32)[np.newaxis, :, :]
            cv2.fillPoly(score_map, shrinked_poly, 1)
            cv2.fillPoly(poly_mask, shrinked_poly, poly_idx + 1)

            # dont care area between margin and inner area
            if this.margin_dont_care:
                dontcare_ratio = 0.
                mask_dontcare = shrink_poly(poly.copy(), r, dontcare_ratio).astype(np.int32)[np.newaxis, :, :]
                cv2.fillPoly(training_mask, mask_dontcare, 0)
                cv2.fillPoly(training_mask, shrinked_poly, 1)

            # if the poly is too small, then ignore it during training
            poly_h = min(np.linalg.norm(poly[0] - poly[3]), np.linalg.norm(poly[1] - poly[2]))
            poly_w = min(np.linalg.norm(poly[0] - poly[1]), np.linalg.norm(poly[2] - poly[3]))
            if min(poly_h, poly_w) < this.min_text_size:
                cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
            if tag:
                cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)

            xy_in_poly = np.argwhere(poly_mask == (poly_idx + 1))

            parallelogram=this.poly_2_para(poly);

            # Drop bad batch.
            if(parallelogram is None):
                return None,None,None,None;

            parallelogram_coord_sum = np.sum(parallelogram, axis=1)
            min_coord_idx = np.argmin(parallelogram_coord_sum)
            parallelogram = parallelogram[
                [min_coord_idx, (min_coord_idx + 1) % 4, (min_coord_idx + 2) % 4, (min_coord_idx + 3) % 4]]

            rectange = rectangle_from_parallelogram(parallelogram)
            rectange, rotate_angle = sort_rectangle(rectange)

            p0_rect, p1_rect, p2_rect, p3_rect = rectange



            fp=xy_in_poly.astype(float)[:,::-1];
            #
            # # geo_map[xy_in_poly[:,0],xy_in_poly[:,1],:4]=neko_points_to_each_line(np.array([p0_rect,p1_rect,p2_rect,p3_rect]),np.array([p1_rect,p2_rect,p3_rect,p0_rect]),fp)
            # eye[:,0]=neko_points_to_line(p0_rect,p1_rect,fp);
            # eye[:,1] = neko_points_to_line(p1_rect, p2_rect, fp);
            # eye[:,2] = neko_points_to_line(p2_rect, p3_rect, fp);
            # eye[:,3] = neko_points_to_line(p3_rect, p0_rect, fp);
            # eye[:,4] = rotate_angle;


            geo_map[xy_in_poly[:,0],xy_in_poly[:,1],0]=neko_points_to_line(p0_rect,p1_rect,fp);
            geo_map[xy_in_poly[:,0],xy_in_poly[:,1],1] = neko_points_to_line(p1_rect, p2_rect, fp);
            geo_map[xy_in_poly[:,0],xy_in_poly[:,1],2] = neko_points_to_line(p2_rect, p3_rect, fp);
            geo_map[xy_in_poly[:,0],xy_in_poly[:,1],3] = neko_points_to_line(p3_rect, p0_rect, fp);
            geo_map[xy_in_poly[:,0],xy_in_poly[:,1],4] = rotate_angle;
            #
            # for y, x in xy_in_poly:
            #     point = np.array([x, y], dtype=np.float32)
            #     # top
            #     g[y, x, 0] = point_dist_to_line(p0_rect, p1_rect, point)
            #     # right
            #     g[y, x, 1] = point_dist_to_line(p1_rect, p2_rect, point)
            #
            #     # down
            #     g[y, x, 2] = point_dist_to_line(p2_rect, p3_rect, point)
            #     # left
            #     g[y, x, 3] = point_dist_to_line(p3_rect, p0_rect, point)
            #     # angle
            #     g[y, x, 4] = rotate_angle
            # eye = geo_map[xy_in_poly[:, 0], xy_in_poly[:, 1]];
            # eye1=g[xy_in_poly[:,0],xy_in_poly[:,1]];
            #
            # g-=geo_map;
            # print(geo_map.max());
            # print(g.max(),g.min());

        return np.expand_dims(score_map[::scale,::scale].astype(np.float32),-1), \
               geo_map[::scale,::scale].astype(np.float32),\
               np.expand_dims(training_mask[::scale,::scale].astype(np.float32),-1), \
               np.expand_dims(poly_mask[::scale,::scale].astype(np.float32),-1);

    # score_map,geo_map,training_mask,poly_mask
    def generate_quad_gt(this,im_size_hw, text_polys, text_tags,scale=4,**kwargs ):
        h, w = im_size_hw
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        score_map = np.zeros((h, w), dtype=np.uint8)
        geo_map = np.zeros((h, w, 8), dtype=np.float32)
        # mask used during traning, to ignore some hard areas
        training_mask = np.ones((h, w), dtype=np.uint8)
        for poly_idx, poly_tag in enumerate(zip(text_polys, text_tags)):
            poly = poly_tag[0]
            tag = poly_tag[1]


            r = [None, None, None, None]
            for i in range(4):
                r[i] = min(np.linalg.norm(poly[i] - poly[(i + 1) % 4]),
                           np.linalg.norm(poly[i] - poly[(i - 1) % 4]))
            # score map
            shrinked_poly = shrink_poly(poly.copy(), r).astype(np.int32)[np.newaxis, :, :]
            cv2.fillPoly(score_map, shrinked_poly, 1)
            cv2.fillPoly(poly_mask, shrinked_poly, poly_idx + 1)

            # dont care area between margin and inner area
            if this.margin_dont_care:
                dontcare_ratio = 0.
                mask_dontcare = shrink_poly(poly.copy(), r, dontcare_ratio).astype(np.int32)[np.newaxis, :, :]
                cv2.fillPoly(training_mask, mask_dontcare, 0)
                cv2.fillPoly(training_mask, shrinked_poly, 1)

            # if the poly is too small, then ignore it during training
            poly_h = min(np.linalg.norm(poly[0] - poly[3]), np.linalg.norm(poly[1] - poly[2]))
            poly_w = min(np.linalg.norm(poly[0] - poly[1]), np.linalg.norm(poly[2] - poly[3]))
            if min(poly_h, poly_w) < this.min_text_size:
                cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
            if tag:
                cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)

            xy_in_poly = np.argwhere(poly_mask == (poly_idx + 1))
            fp=xy_in_poly.astype(float)[:,::-1];

            for i in range(4):
                geo_map[xy_in_poly[:,0],xy_in_poly[:,1],i*2:i*2+2]=(poly[i]-fp)


        return np.expand_dims(score_map[::scale,::scale].astype(np.float32),-1), \
               geo_map[::scale,::scale].astype(np.float32),\
               np.expand_dims(training_mask[::scale,::scale].astype(np.float32),-1), \
               np.expand_dims(poly_mask[::scale,::scale].astype(np.float32),-1);


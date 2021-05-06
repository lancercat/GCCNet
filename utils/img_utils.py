import glob;
import os;
import numpy as np
import csv;
import cv2;
import cv2, csv
import time
import os
import numpy as np
import math

from utils.geometry_utils import welf_crop_textline

class HJB_rotate:
    CV_PI = 3.14159265
    @staticmethod
    def rotate_a_point(pt, center, degree, movSize=[0, 0], scale=1):
        angle = degree * HJB_rotate.CV_PI / 180.0
        alpha = math.cos(angle)
        beta = math.sin(angle)
        x = (pt[0] - center[0]) * scale
        y = (pt[1] - center[1]) * scale
        dstPt = [0, 0]
        dstPt[0] = round(x * alpha + y * beta + center[0] + movSize[0])
        dstPt[1] = round(-x * beta + y * alpha + center[1] + movSize[1])
        return dstPt
    @staticmethod
    def rotate_poly(polys, center, degree, movSize=[0, 0], scale=1):
        det_polys = []
        for p in polys:
            det_polys.append(HJB_rotate.rotate_a_point(p, center, degree, movSize, scale))
        return det_polys
    @staticmethod
    def shift(size, degree):
        angle = degree * HJB_rotate.CV_PI / 180.0
        width = size[0]
        height = size[1]

        alpha = math.cos(angle)
        beta = math.sin(angle)
        new_width = (int)(width * math.fabs(alpha) + height * math.fabs(beta))
        new_height = (int)(width * math.fabs(beta) + height * math.fabs(alpha))

        size = [new_width, new_height]
        return size
    @staticmethod
    def rotate_image_and_gt(src_im, src_rects, degree, scale=1):
        degree2 = -degree
        W = src_im.shape[1]
        H = src_im.shape[0]
        center = (W // 2, H // 2)
        newSize = HJB_rotate.shift([W * scale, H * scale], degree2);
        # dst = Mat(newSize, CV_32FC3)
        M = cv2.getRotationMatrix2D(center, degree2, scale)
        M[0, 2] += (int)((newSize[0] - W) / 2);
        M[1, 2] += (int)((newSize[1] - H) / 2);
        dst_im = cv2.warpAffine(src_im, M, (newSize[0], newSize[1]))
        # warpAffine(src_im, dst, M, cvSize(newSize.width, newSize.height), CV_INTER_LINEAR, BORDER_CONSTANT,sc);

        movSize = [int((newSize[0] - W) / 2), int((newSize[1] - H) / 2)]
        dst_rects = []
        for rs in src_rects:
            dst_rects.append(HJB_rotate.rotate_poly(rs, center, degree2, movSize, scale))
        return dst_im, dst_rects


def get_images(path):
    files = []
    for ext in ['jpg', 'png', 'jpeg', 'JPG']:
        files.extend(glob.glob(
            os.path.join(path, '*.{}'.format(ext))))
    return files


def get_imdb(imdb):
    files = []
    with open(imdb, 'r') as imdb_f:
        for img in imdb_f:
            files.append(img.strip())
    return files

def rotate_image90(im, polys):
    nim=np.transpose(im,[1,0,2]);
    npolys=np.zeros_like(polys);
    npolys[:,:,0]=polys[:,:,1];
    npolys[:,:,1]=im.shape[0]-polys[:,:,0];
    return nim,npolys;

def get_crop_meta(h,w,polys,max_tries,min_crop_side_ratio,crop_background):
    pad_h = h // 10
    pad_w = w // 10
    h_array = np.zeros((h + pad_h * 2), dtype=np.int32)
    w_array = np.zeros((w + pad_w * 2), dtype=np.int32)
    for poly in polys:
        poly = np.round(poly, decimals=0).astype(np.int32)
        minx = np.min(poly[:, 0])
        maxx = np.max(poly[:, 0])
        w_array[minx + pad_w:maxx + pad_w] = 1
        miny = np.min(poly[:, 1])
        maxy = np.max(poly[:, 1])
        h_array[miny + pad_h:maxy + pad_h] = 1
    # ensure the cropped area not across a text
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]
    if len(h_axis) == 0 or len(w_axis) == 0:
        return None,None;

    for i in range(max_tries):
        xx = np.random.choice(w_axis, size=2)
        xmin = np.min(xx) - pad_w
        xmax = np.max(xx) - pad_w
        xmin = np.clip(xmin, 0, w - 1)
        xmax = np.clip(xmax, 0, w - 1)
        yy = np.random.choice(h_axis, size=2)
        ymin = np.min(yy) - pad_h
        ymax = np.max(yy) - pad_h
        ymin = np.clip(ymin, 0, h - 1)
        ymax = np.clip(ymax, 0, h - 1)
        if xmax - xmin < min_crop_side_ratio * w or ymax - ymin < min_crop_side_ratio * h:
            # area too small
            continue
        if polys.shape[0] != 0:
            poly_axis_in_area = (polys[:, :, 0] >= xmin) & (polys[:, :, 0] <= xmax) \
                                & (polys[:, :, 1] >= ymin) & (polys[:, :, 1] <= ymax)
            selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]
        else:
            selected_polys = [];

        if len(selected_polys) == 0:
            # no text in this area
            if crop_background:
                return [ymin,ymax,xmin,xmax],selected_polys;
            else:
                continue
        return  [ymin,ymax,xmin,xmax],selected_polys

    return None,None;


def crop_area(im, polys, tags, crop_background=False, max_tries=50,min_crop_side_ratio=0.1):
    '''
    make random crop from the input image
    :param im:
    :param polys:
    :param tags:
    :param crop_background:
    :param max_tries:
    :return:
    '''
    h, w, _ = im.shape
    loc,selected_polys=get_crop_meta(h,w,polys,max_tries,min_crop_side_ratio,crop_background);
    if(loc is None):
        return im,polys,tags;
    [ymin,ymax,xmin,xmax]=loc;
    im = im[ymin:ymax+1, xmin:xmax+1, :]
    if(polys.shape[0]):
        polys = polys[selected_polys]
        tags = tags[selected_polys]
        polys[:, :, 0] -= xmin
        polys[:, :, 1] -= ymin
    return im, polys, tags



def crop_area_rec(im, polys, tags,texts, crop_background=False, max_tries=50,min_crop_side_ratio=0.1):
    '''
       make random crop from the input image
       :param im:
       :param polys:
       :param tags:
       :param crop_background:
       :param max_tries:
       :return:
       '''
    h, w, _ = im.shape
    loc, selected_polys = get_crop_meta(h, w, polys, max_tries, min_crop_side_ratio, crop_background);
    if (loc is None):
        return im, polys, tags;
    [ymin, ymax, xmin, xmax] = loc;
    im = im[ymin:ymax + 1, xmin:xmax + 1, :]
    polys = polys[selected_polys]
    tags = tags[selected_polys]
    texts=texts[selected_polys];
    polys[:, :, 0] -= xmin
    polys[:, :, 1] -= ymin
    return im, polys, tags,texts;


def resize_image(im, max_side_len=2400, re_ratio=1.):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    resize_h = int(h * re_ratio)
    resize_w = int(w * re_ratio)

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / h if h >w else float(max_side_len) / w
    else:
        ratio = re_ratio

    resize_h = int(h * ratio)
    resize_w = int(w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 ) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 ) * 32
    if(resize_h==0 or resize_w==0):
        return None,(9,9);
    im = cv2.resize(im, (int(resize_w), int(resize_h)))
    print(resize_h,resize_w);
    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)



def resize_image_min_edge(im, max_side_len=1366, min_edge=720.0):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h
    #print("mew",min_edge,min(w,h));

    re_ratio=min_edge/min(w,h);
    resize_h = int(h * re_ratio)
    resize_w = int(w * re_ratio)
    # print("mew",resize_h,resize_w);

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / h if h > w else float(max_side_len) / w
    else:
        ratio = re_ratio

    # print("meew",ratio,re_ratio);

    resize_h = int(h * ratio)
    resize_w = int(w * ratio)
    # print("meew",resize_w,resize_h);

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    if(resize_h==0 or resize_w==0):
        return None,(9,9);
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)
    # print("meeew",resize_w,resize_h);

    return im, (ratio_h, ratio_w)

def resize_image_min_edge_nkar(im, max_side_len=1366, min_edge=720.0):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape



    re_ratio=min_edge/min(w,h);
    resize_h = int(h * re_ratio)
    resize_w = int(w * re_ratio)
    # print("mew",resize_h,resize_w);

    # limit the max side
    if resize_h > max_side_len:
        resize_h = max_side_len;
    if resize_w > max_side_len:
        resize_w = max_side_len;


    resize_h = resize_h if resize_h % 32 == 0 else int(resize_h / 32 ) * 32
    resize_w = resize_w if resize_w % 32 == 0 else int(resize_w / 32) * 32
    if(resize_h<min_edge):
        resize_h=min_edge;
    if(resize_w<min_edge):
        resize_w=min_edge;
    if(resize_h<min_edge or resize_w<min_edge):
        return None,(9,9);

    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)
    # print("meeew",resize_w,resize_h);

    return im, (ratio_h, ratio_w)


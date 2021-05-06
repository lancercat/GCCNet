# from utils.geometry_utils import point_dist_to_line;
import cv2;
import numpy as np;
import random;
from utils.geometry_utils import check_and_validate_polys;
from utils.img_utils import crop_area,crop_area_rec,HJB_rotate,rotate_image90;

#TODO[v5] from desc to desc
class augumenter:
    @staticmethod
    def get_zsx_augumented(raw_entry, input_size=512,
                       background_ratio=0. / 8,
                       random_scale=np.array([0.5, 1, 2.0, 1.5]),
                       mean=np.array([127, 127, 127]), var=np.array([127, 127, 127]),rotate_range=None):
        pass;

    @staticmethod
    def get_east_augumented(raw_entry, input_size,
                       background_ratio,
                       random_scale,
                       mean, var,rotate_range):
        # print(im_fn)
        im = (cv2.imread(raw_entry.im_path).astype(np.float32) - mean) / var;
        # print(im_fn)
        h, w, _ = im.shape;
        text_polys = np.array(raw_entry.boxes, dtype=np.float32);
        text_tags = np.array(raw_entry.det_dcs, dtype=np.bool);
        text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (h, w));
        if(rotate_range is not None):
            angle = random.uniform(rotate_range[0], rotate_range[1]);
            im, text_polys = HJB_rotate.rotate_image_and_gt(im, text_polys, angle)

        text_polys = np.array(text_polys, dtype=np.float32)
        rd_scale = np.random.choice(random_scale)
        im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
        text_polys *= rd_scale
        # print(rd_scale)
        # random crop a area from image
        if np.random.rand() < background_ratio:
            # crop background
            im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=True)
            if text_polys.shape[0] > 0:
                # cannot find background
                return None, None, None;
            # pad and resize image
            new_h, new_w, _ = im.shape
            max_h_w_i = np.max([new_h, new_w, input_size])
            im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
            im_padded[:new_h, :new_w, :] = im.copy()
            im = cv2.resize(im_padded, dsize=(input_size, input_size))
        else:
            im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=False)
            if text_polys.shape[0] == 0:
                return None, None, None;
            h, w, _ = im.shape

            # pad the image to the training input size or the longer side of image
            new_h, new_w, _ = im.shape
            max_h_w_i = np.max([new_h, new_w, input_size])
            im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.float32)
            im_padded[:new_h, :new_w, :] = im.copy()
            # im = im_padded
            # resize the image to input size
            new_h, new_w, _ = im_padded.shape
            resize_h = input_size
            resize_w = input_size
            im_padded = cv2.resize(im_padded, dsize=(resize_w, resize_h))
            resize_ratio_3_x = resize_w / float(new_w)
            resize_ratio_3_y = resize_h / float(new_h)
            text_polys[:, :, 0] *= resize_ratio_3_x
            text_polys[:, :, 1] *= resize_ratio_3_y
        return im_padded, text_polys, text_tags;
    @staticmethod
    def get_rec_east_augumented(raw_entry, input_size=512,
                       background_ratio=0. / 8,
                       random_scale=np.array([0.5, 1, 2.0, 1.5]),
                       mean=np.array([127, 127, 127]), var=np.array([127, 127, 127])):
        # print(im_fn)
        im = (cv2.imread(raw_entry.im_path).astype(np.float32) - mean) / var;
        # print(im_fn)
        h, w, _ = im.shape;
        text_polys = np.array(raw_entry.boxes, dtype=np.float32);
        texts=raw_entry.texts;
        text_tags = np.array(raw_entry.det_dcs, dtype=np.bool);
        text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (h, w))
        rd_scale = np.random.choice(random_scale)
        im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
        text_polys *= rd_scale
        # print(rd_scale)
        # random crop a area from image
        if np.random.rand() < background_ratio:
            # crop background
            im, text_polys, text_tags,texts = crop_area_rec(im, text_polys, text_tags,texts, crop_background=True)
            if text_polys.shape[0] > 0:
                # cannot find background
                return None, None, None;
            # pad and resize image
            new_h, new_w, _ = im.shape
            max_h_w_i = np.max([new_h, new_w, input_size])
            im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
            im_padded[:new_h, :new_w, :] = im.copy()
            im = cv2.resize(im_padded, dsize=(input_size, input_size))
        else:
            im, text_polys, text_tags,texts = crop_area_rec(im, text_polys, text_tags,texts, crop_background=False)
            if text_polys.shape[0] == 0:
                return None, None, None;
            h, w, _ = im.shape

            # pad the image to the training input size or the longer side of image
            new_h, new_w, _ = im.shape
            max_h_w_i = np.max([new_h, new_w, input_size])
            im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.float32)
            im_padded[:new_h, :new_w, :] = im.copy()
            im = im_padded
            # resize the image to input size
            new_h, new_w, _ = im.shape
            resize_h = input_size
            resize_w = input_size
            im = cv2.resize(im, dsize=(resize_w, resize_h))
            resize_ratio_3_x = resize_w / float(new_w)
            resize_ratio_3_y = resize_h / float(new_h)
            text_polys[:, :, 0] *= resize_ratio_3_x
            text_polys[:, :, 1] *= resize_ratio_3_y
        return im, text_polys, text_tags;

    @staticmethod
    def get_rotated_augumented(raw_entry, input_size=512,
                            background_ratio=0. / 8,
                            random_scale=np.array([0.5, 1, 2.0, 1.5]),
                            mean=np.array([127, 127, 127]), var=np.array([127, 127, 127]),magic=[0,0,0,0,0,1,1,1,2,2,3,3,3]):
        # print(im_fn)
        im = (cv2.imread(raw_entry.im_path).astype(np.float32) - mean) / var;
        # print(im_fn)
        h, w, _ = im.shape;
        text_polys = np.array(raw_entry.boxes, dtype=np.float32);
        text_tags = np.array(raw_entry.det_dcs, dtype=np.bool);
        text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (h, w))
        rd_scale = np.random.choice(random_scale)
        im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
        text_polys *= rd_scale
        # print(rd_scale)
        # random crop a area from image
        if np.random.rand() < background_ratio:
            # crop background

            im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=True)


            ntimes=random.choice(magic);
            for i in range(ntimes):
                im,text_polys=rotate_image90(im,text_polys)

            if text_polys.shape[0] > 0:
                # cannot find background
                return None, None, None;
            # pad and resize image
            new_h, new_w, _ = im.shape
            max_h_w_i = np.max([new_h, new_w, input_size])
            im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
            im_padded[:new_h, :new_w, :] = im.copy()
            im = cv2.resize(im_padded, dsize=(input_size, input_size))
        else:

            im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=False)

            ntimes = random.choice(magic);
            for i in range(ntimes):
                im, text_polys = rotate_image90(im, text_polys)

            if text_polys.shape[0] == 0:
                return None, None, None;
            h, w, _ = im.shape

            # pad the image to the training input size or the longer side of image
            new_h, new_w, _ = im.shape
            max_h_w_i = np.max([new_h, new_w, input_size])
            im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.float32)
            im_padded[:new_h, :new_w, :] = im.copy()
            im = im_padded
            # resize the image to input size
            new_h, new_w, _ = im.shape
            resize_h = input_size
            resize_w = input_size
            im = cv2.resize(im, dsize=(resize_w, resize_h))
            resize_ratio_3_x = resize_w / float(new_w)
            resize_ratio_3_y = resize_h / float(new_h)
            text_polys[:, :, 0] *= resize_ratio_3_x
            text_polys[:, :, 1] *= resize_ratio_3_y
        return im, text_polys, text_tags;

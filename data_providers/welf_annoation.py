import  numpy as np;
import csv;
import os ;
from utils.det_box_cvtr import rbox2points as label_convert
def load_annoatation(gt_path, gt_name):
    '''
    load annotation from the text file
    :param gt_path, gt_name:
    :return:
    '''
    text_polys = []
    text_tags = []
    txt_fn = os.path.join(gt_path, 'gt_'+gt_name+'.txt')
    #print(txt_fn)
    if not os.path.exists(txt_fn):
        print('text file {} does not exists'.format(txt_fn))
        return np.array(text_polys, dtype=np.float32)
    with open(txt_fn, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            label = line[-1]
            # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]

            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
            text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            if label == '*' or label == '###':
                text_tags.append(True)
            else:
                text_tags.append(False)
        return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)



def load_annoatation_imdb(gt_path, gt_name):
    '''
    load annotation from the text file
    :param gt_path, gt_name:
    :return:
    '''
    text_polys = []
    text_tags = []
    txt_fn = os.path.join(gt_path, 'gt_'+gt_name+'.txt');

    #print(txt_fn)
    if not os.path.exists(txt_fn):
        print('text file {} does not exists'.format(txt_fn))
        return np.array(text_polys, dtype=np.float32)
    with open(txt_fn, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            boxes = label_convert(map(float, line[:5]))
            text_polys.append(boxes.reshape((4, 2)).tolist())
            label = 'xxx'
            if label == '*' or label == '###':
                text_tags.append(True)
            else:
                text_tags.append(False)
        return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)



def read_synthtext(gt_path, gt_name):
    '''
    load annotation from the text file
    :param gt_path, gt_name:
    :return:
    '''
    text_polys = []
    text_tags = []
    txt_fn = os.path.join(gt_path, gt_name+'.txt')
    #print(txt_fn)
    if not os.path.exists(txt_fn):
        print('text file {} does not exists'.format(txt_fn))
        return np.array(text_polys, dtype=np.float32)
    with open(txt_fn, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, line[:8])
            text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            label = line[-1]
            if label == '*' or label == '###':
                text_tags.append(True)
            else:
                text_tags.append(False)
        return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)




def get_data_reader(imdb_name):
    reader_factory = {'icdar': load_annoatation,
                      'synthtext': read_synthtext,
                      'ad': load_annoatation_imdb}
    return reader_factory[imdb_name]

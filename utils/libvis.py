# encoding=utf-8
import cv2
import pygame
import random
from pygame import  freetype
import numpy as np
#from powerline.bindings.zsh import string
#from __builtin__ import open
from pygame.locals import *
import sys
import cchardet
from os import listdir
from os.path import isfile, join
import codecs
#import freetype
from utils.simple_vis import det_vis;
#/home/lasercat/cat/py-faster-rcnn-master/lib/rpn_tex/Imageaddfont.py
class font_cfg:
    def __init__(this):
        this.double_max_font_ratio = 0.4;
        this.double_min_font_ratio = 0.1;
        this.possible_styles = [pygame.freetype.STYLE_NORMAL, pygame.freetype.STYLE_UNDERLINE,
                                pygame.freetype.STYLE_OBLIQUE, pygame.freetype.STYLE_STRONG,
                                pygame.freetype.STYLE_WIDE, pygame.freetype.STYLE_DEFAULT];


class render:
    def __init__(self):
        pygame.freetype.init();

    def cvimage_to_pygame(this,image):
        """Convert cvimage into a pygame image"""
        return pygame.image.frombuffer(image.astype(np.uint8).tostring(), image.shape[1::-1],
                                       "RGB")

    def Rect_add_font(this,font_font, string_text, Mat_im, int_x, int_y,bgcolor=None):

        bounds = font_font.get_rect(string_text);
        bounds.x = int_x;
        bounds.y = int_y;
        baz = this.cvimage_to_pygame(Mat_im)
        font_font.render_to(baz, bounds, string_text,bgcolor=bgcolor);

        Mat_im = pygame.surfarray.pixels3d(baz);
        Mat_im = Mat_im.transpose(1, 0, 2);

        # cv2.imshow("baz",Mat_im);
        # cv2.waitKey(30);

        return Mat_im, bounds;

    def list_rect_add_font_multiline(this,font, string_str, Mat_im, int_x=10, int_y=10, int_line_width=200,
                                     int_line_space=10):
        # font.kerning = True
        # font.strength = 0.5
        lines = [];
        real_height = [];

        line_beg = 0;
        line_end = 1;

        while line_beg < len(string_str):
            string_line = "";
            while line_end < len(string_str):
                string_line_lgr = string_str[line_beg:line_end];
                bounds = font.get_rect(string_line_lgr);

                if (bounds.width < int_line_width):
                    string_line = string_line_lgr;
                    line_end += 1;
                else:
                    break
            line_beg = line_end;
            if (len(string_line) < 1):
                continue;
            real_height.append(max(font.get_rect(string_line).height, font.size));
            lines.append(string_line)
            line_end += 1;
            bounds = font.get_rect(string_line);
        res = [];

        for i in range(0, len(lines)):
            if (int_y + real_height[i] > Mat_im.shape[0]):
                break;
            Mat_im, gt = this.Rect_add_font(font, lines[i], Mat_im, int_x, int_y);

            res.append(gt);
            int_y += real_height[i] + int_line_space;
        return Mat_im, res

    def write(Mat_im, list_rect_gt, string_path, int_fake_id):
        #
        cv2.imwrite(string_path + str(int_fake_id) + ".jpg", Mat_im);
        file_fp = open(string_path + str(int_fake_id) + ".txt", "w");
        for rect_gt in list_rect_gt:
            file_fp.writelines(str(rect_gt.x) + " " + \
                               str(rect_gt.y) + " " + \
                               str(rect_gt.x + rect_gt.width) + " " + \
                               str(rect_gt.y + rect_gt.height) + "\n");

    # pick some random image from string_im_source
    # add

class e2e_vis:

    def __init__(this,font):
        this.render=render();
        this.font=pygame.freetype.Font(font,12);
        this.bgcolor=(128,255,255,255);
        this.font.fgcolor=(255,0,0);
    def vis_word(this,im,box,res):
        x,y,w,h=cv2.boundingRect(box.astype(int).reshape(-1,2));
        return this.render.Rect_add_font(this.font,res,im,x,y,this.bgcolor);

    def vis(this,im,boxes,ress):

        for i in range(len(ress)):
            cv2.polylines(im, [boxes[i].astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0),
                          thickness=1);
            im,_=this.vis_word(im,boxes[i],ress[i]);
        return im;
    def vis_wcolor(this,im,boxes,ress,cls,colors):
        for i in range(len(ress)):
            color=colors[cls[i]];
            cv2.polylines(im, [boxes[i].astype(np.int32).reshape((-1, 1, 2))], True, color=color,
                          thickness=1);
            im,_=this.vis_word(im,boxes[i],ress[i]);
        return im;
    def vis_det(this,im,boxes):
        ress=[];
        for i in boxes.shape[0]:
            ress.append("");
        return this.vis(im,boxes,ress);

# Font_font.style=pygame.freetype.STYLE_OBLIQUE;
class monkey:
    cfg = {}
    cfg['font_size_ratio'] = 0.1
    cfg['font_size_min'] = 30
    cfg['font_size_max'] = 100

    def make_random_color(this,style=0):
        if style == 0:
            c = random.randint(200, 255)
            f = random.randint(0, 8)
            if f >= 2:
                return (c, c, c)
            else:
                return (255 - c, 255 - c, 255 - c)

    # pick some random image from string_im_source
    # add


    def Font_random_font(this,string_font_file, int_im_min_edge=600):
        font_cfg_somecfg = font_cfg();
        int_size = random.randint(int(int_im_min_edge * font_cfg_somecfg.double_min_font_ratio),
                                  int(int_im_min_edge * font_cfg_somecfg.double_max_font_ratio));
        Font_font = pygame.freetype.Font(string_font_file, int_size);
        Font_font.style = random.choice(font_cfg_somecfg.possible_styles);
        # Font_font.fgcolor = (random.randint(0,255), random.randint(0,255), random.randint(0,255));
        Font_font.fgcolor = this.make_random_color()
        return Font_font;

    def Font_random_font_H(this,string_font_file, int_im_min_edge=600):
        font_cfg_somecfg = font_cfg();
        int_size = random.randint(min(this.cfg['font_size_min'], int_im_min_edge),
                                  min(this.cfg['font_size_max'], int_im_min_edge));
        if (int_im_min_edge < 600):
            int_size = int_im_min_edge * this.cfg['font_size_ratio'];
        Font_font = pygame.freetype.Font(string_font_file, int_size);
        Font_font.style = random.choice(font_cfg_somecfg.possible_styles);
        # Font_font.fgcolor = (random.randint(0,255), random.randint(0,255), random.randint(0,255));
        Font_font.fgcolor = this.make_random_color()
        return Font_font;

    def string_random_string(this,string_src):
        int_length = len(string_src);
        int_substr_len = random.randint(3, min(300, int_length));
        int_start_pos = random.randint(0, int_length - int_substr_len);
        int_end_pos = int_start_pos + int_substr_len;
        return string_src[int_start_pos:int_end_pos];



    def string_load_and_convert_to_utf8(this,string_file_name):
        FILE_fp = open(string_file_name, "rb");
        string_raw_bytes = FILE_fp.read();
        string_encoding = cchardet.detect(string_raw_bytes)["encoding"].encode("ASCII");
        FILE_fp.close();
        FILE_fp = codecs.open(string_file_name, "r", encoding=string_encoding);
        string_s = FILE_fp.read();
        return string_s;

    def void_random_generator(this,int_cnt, string_font_path, string_text_source, string_im_source, string_dst):
        list_text_files = [join(string_text_source, f) for f in listdir(string_text_source) if
                           isfile(join(string_text_source, f))];
        list_font_files = [join(string_font_path, f) for f in listdir(string_font_path) if
                           isfile(join(string_font_path, f))];
        list_images = [join(string_im_source, f) for f in listdir(string_im_source) if
                       isfile(join(string_im_source, f))]

        list_text_pieces = [];

        for string_file_name in list_text_files:
            list_text_pieces.append(this.string_load_and_convert_to_utf8(string_file_name));
        for i in range(0, int_cnt):
            string_s = this.string_random_string(string_file_name);
            im = cv2.imread(random.choice(list_images));

            # Font_font=Font_random_font(random.choice(list_font_files),min(im.shape[1],im.shape[0]));
            Font_font = this.Font_random_font_H(random.choice(list_font_files), min(im.shape[1], im.shape[0]));

            string_raw = this.string_random_string(random.choice(list_text_pieces));
            int_font_size = Font_font.get_sized_height();
            int_line_width = random.randint(int_font_size, im.shape[1]);

            int_x = random.randint(0, im.shape[1] - int_line_width);
            int_y = random.randint(0, im.shape[0] - int_font_size);
            int_line_space = random.randint(0, int_font_size * 2);
            im, gt = this.list_rect_add_font_multiline(Font_font, string_raw, im, int_x, int_y, int_line_width,
                                                  int_line_space);
            this.write(im, gt, string_dst, i + 230);



    def __init__(this,path_base):
        this.string_path_base = path_base;
        this.string_path_font = this.string_path_base + "/fonts";
        # this.string_path_image = this.string_path_base + "/image_src";
        # this.string_path_text = this.string_path_base + "/text_src";
        # this.string_path_target = this.string_path_base + "/target/";
        pygame.init();

    # void_random_generator(10, string_path_font, string_path_text, string_path_image, string_path_target);

# text = u"中文名称 歼-20 英文名称 Chengdu J-20 服役时间 2017年（预计） 定型时间 2015年 国 家 中华人民共和国 设计单位 中航工业成都飞机设计研究所 制造单位";
# im = cv2.imread("/home/lasercat/cat/py-faster-rcnn-master/data/manual_insertinon/05300004787906131900486419856.jpg")
# pygame.init()
# surface = pygame.surfarray.make_surface(im.swapaxes(0,1))
#
# Font_font =pygame.freetype.Font("/home/lasercat/cat/py-faster-rcnn-master/data/manual_insertinon/KaiTi_GB2312.ttf",30);
# Font_font.style=pygame.freetype.STYLE_OBLIQUE;
# Font_font.fgcolor=(0,0,0);
# im,gt=list_rect_add_font_multiline(Font_font,text,im,int_line_width=500);
# write(im,gt,"/home/lasercat/cat/py-faster-rcnn-master/data/manual_insertinon/",230);
#/home/lasercat/cat/py-faster-rcnn-master/lib/rpn_tex/Imageaddfont.py

import numpy as np
import cv2
import random
import os
#import this as this

class kot_translator:
    def __init__(this,dict_file):
        this.word_dict = []

        this.word_dict.append('END');
        this.word_dict.append('STA');
        this.word_dict.append('PAD');
        this.word_dict.append("UK");
        for w in open(dict_file, 'r'):
            # print(w)
            #char = w.replace('\n', '')
            try:
                char = w.split()[1].strip()
                this.word_dict.append(char);
            except:
                print(w);
                pass;

            # print(char)
        # for w in open(this.symbol_dict_file, 'r'):
        #    char = w.strip()
        #    this.word_dict.append(char.decode('utf-8'))
        # import codecs
        # for w in codecs.open(this.ethnic_dict_file, 'r', encoding='utf-8'):
        #    line = w.strip()#.decode('utf-8')
        # print(line,len(line))
        #    for c in line:
        # print(c)
        #        if c not in this.word_dict:
        # print(c.encode('utf-8'))
        #            this.word_dict.append(c)
        # print(c.decode('utf-8'))
        # print(char)
        print(len(this.word_dict))
        # for w in this.word_dict:
        #    print(w.encode('utf-8'))

    def get_dict(this):
        return this.word_dict

    def get_label(this,s):
        if s in this.word_dict:
            return this.word_dict.index(s)
        else:
            return this.word_dict.index("UK"); #-1

    def show_text(this,x):
        text=''
        #text = ''.join([index2char[x[0, i]] for i in range(x.shape[1])])
        for i in range(x.shape[0]):
            if this.index2char(x[i])=='END':
                break
            text = text+ this.index2char(x[i]);
        #end_index = text.find('END')
        #text = text[:end_index]
        return text;

    def show_text_batch(this, x):
        text=''
        #text = ''.join([index2char[x[0, i]] for i in range(x.shape[1])])
        for i in range(x.shape[0]):
            text+=this.show_text(x[i]);#end_index = text.find('END')
        #text = text[:end_index]
        text+='\n';
        return text

    def symbol_cnt(this):
        return len(this.word_dict);

    def index2char(this,x):
        return this.word_dict[int(x)]

    def char2index(this,x):
        return this.get_label(x)

    def getstr(this,l,max_text):   # not used
        result=""
        for i in l:
            if this.index2char(i)== 'END':
                break
            if i <max_text and i>-1:
                result+=this.index2char(i)
        return result

    def get_line_accuracy_lst(this,x, y):
        ed_lst = []
        for j in range(x.shape[0]):
            x_text = ''.join([ this.index2char(x[j,i]) for i in range(x.shape[1]) ])
            end_index = x_text.find('END')
            x_text = x_text[:end_index]
            y_text = ''.join([ this.index2char(y[j,i]) for i in range(y.shape[1]) ])
            end_index = y_text.find('END')
            y_text =  y_text[:end_index]

            if x_text == y_text:
                ed_lst.append(1.0)
            else:
                ed_lst.append(0.0)
        return ed_lst

    def translate(this, txt):
        temp_txt=[this.get_label(c) for c in txt];
        temp_txt.append(this.get_label('END'));
        temp_txt=np.array(temp_txt).astype(int);
        valid = (temp_txt != this.get_label('#')).astype(float);
        return temp_txt,valid;

    def get_pad(this,pad_len):
        temp_txt=[];
        for i in range(pad_len):
            temp_txt.append(this.get_label('PAD'));
        valid=np.zeros(temp_txt,float);
        return temp_txt,valid;





class translator_t:
    def __init__(this,dict_file):
        this.word_dict = []

        for w in open(dict_file, 'r'):
            # print(w)
            char = w.replace('\n', '')
            # char = w.split()[1].strip()
            this.word_dict.append(char)
            # print(char)
        # for w in open(this.symbol_dict_file, 'r'):
        #    char = w.strip()
        #    this.word_dict.append(char.decode('utf-8'))
        # import codecs
        # for w in codecs.open(this.ethnic_dict_file, 'r', encoding='utf-8'):
        #    line = w.strip()#.decode('utf-8')
        # print(line,len(line))
        #    for c in line:
        # print(c)
        #        if c not in this.word_dict:
        # print(c.encode('utf-8'))
        #            this.word_dict.append(c)
        # print(c.decode('utf-8'))
        # print(char)
        # this.word_dict.append('STA')
        # this.word_dict.append('END')
        # this.word_dict.append('PAD')

        print(len(this.word_dict))
        # for w in this.word_dict:
        #    print(w.encode('utf-8'))

    def get_dict(this):
        return this.word_dict

    def get_label(this,s):
        if s in this.word_dict:
            return this.word_dict.index(s)
        else:
            return this.word_dict.index("#"); #-1


    def show_text(this,x):
        text=''
        #text = ''.join([index2char[x[0, i]] for i in range(x.shape[1])])
        for i in range(x.shape[1]):
            if this.index2char(x[0,i])=='END':
                break
            text = text+ this.index2char(x[0,i]);
        #end_index = text.find('END')
        #text = text[:end_index]
        return text
    def symbol_cnt(this):
        return len(this.word_dict);

    def index2char(this,x):
        return this.word_dict[int(x)]

    def char2index(this,x):
        return this.get_label(x)

    def getstr(this,l,max_text):   # not used
        result=""
        for i in l:
            if this.index2char(i)== 'END':
                break
            if i <max_text and i>-1:
                result+=this.index2char(i)
        return result

    def get_line_accuracy_lst(this,x, y):
        ed_lst = []
        for j in range(x.shape[0]):
            x_text = ''.join([ this.index2char(x[j,i]) for i in range(x.shape[1]) ]);
            end_index = x_text.find('END')
            x_text = x_text[:end_index]
            y_text = ''.join([ this.index2char(y[j,i]) for i in range(y.shape[1]) ])
            end_index = y_text.find('END')
            y_text =  y_text[:end_index]

            if x_text == y_text:
                ed_lst.append(1.0)
            else:
                ed_lst.append(0.0)
        return ed_lst

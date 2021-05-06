# TODO [v4] Coming in v4
from utils.translator import kot_translator
import numpy as np;
class kot_libkotocr:
    def __init__(this,config):
        # different instance shall hold their own translators.
        this.translator=kot_translator(config.get(str,"dict_path"));

    def fancy_traslate(this,txt):
        temp_txt,val=this.translator.translate(txt);
        return np.array(temp_txt), np.array(val);

    def generate_gt(this,txt):
        temp_txt, mask = this.fancy_traslate(txt);
        return temp_txt.astype(int),mask;



class kot_libtaocr:
    def __init__(this,config):
        # different instance shall hold their own translators.
        this.translator=kot_translator(config.get(str,"dict_path"));

    def fancy_traslate(this,txt,max_len):
        temp_txt,val=this.translator.translate(txt);
        pad_len = max_len - len(temp_txt);
        pad,pad_val=this.translator.get_pad(pad_len);
        temp_txt=np.concatenate([temp_txt,pad],0);
        mask=np.concatenate([val,pad_val],0);
        return temp_txt, mask;

    def generate_gt(this,txt,max_len):
        temp_txt, mask = this.fancy_traslate( txt, max_len);
        not_pad = (temp_txt != this.translator.get_label('PAD'))
        not_end = (temp_txt != this.translator.get_label('END'))
        mask2 = np.logical_and(not_pad, not_end).astype(int)
        return temp_txt.astype(int),mask,mask2;


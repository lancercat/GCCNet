import copy;
import json;

class ConfigDict(dict):
    def __init__(self, origin_dict, **kw):
        super(ConfigDict, self).__init__(**kw)
        for k,w in origin_dict.items():
            self[k] = w



class cat_config:

    def __init__(this,dict=None):
        if dict is None:
            this.dict={};
        else:
            this.dict=dict;
        this.child_dict={};
        this.children_index=[];
    def set_opt_item(this,dict,key,dft):
        this.dict[key]=dict.get(key,dft);

    def set_item(this,dict,key):
        this.dict[key]=dict[key];
    def set_item_wdef(this,dict,key,default):
        this.dict[key]=dict.get(key,default);

    def set_list_wdef(this, dict, key, default):
        this.set_list(key,dict.get(key, default));

    def set(this,key,value):
        this.dict[key]=value;

    def merge_default(this,default):
        for i in default.dict.keys:
            if i not  in this.dict:
                this.dict[i]=default.dict[i];
        for i in default.child_dict.keys:
            if i in this.child_dict:
                this.child_dict[i].merge_default(default.child_dict[i]);
            else:
                this.child_dict[i]=default.child_dict;

    def try_get(this, type, key, default=None):
        if(key in this.dict):
            return type(this.dict[key]);
        return default;

    def get(this, type, key):
        if(this.dict[key] == 'None' or this.dict[key] is None):
            return None;
        return type(this.dict[key]);
    def set_list(this,key,values):
        sl=[];
        for  i in values:
            sl.append(str(i));
        this.dict[key]=json.dumps(sl);

    def get_list(this,type,key):
        sl =this.get(str,key);
        sl=json.loads(sl);
        ret=[];
        for i in  sl:
            ret.append(type(i));
        return ret;

    def try_get_list(this, type, key, default):
        if (key in this.dict):
            return this.get_list(type,key);
        return default;

    def get_child(this,key):
        return this.child_dict[key];
    def add_child(this,key,config):
        this.child_dict[key]=copy.deepcopy(config);
        this.children_index.append(key);

    def children_names(this):
        return this.children_index.copy();

    def __str__(this):
        ret=""
        for i in this.dict.keys():
            ret+=this._get_as_xml(i);
        return ret;
    def copy(this):
        ret=cat_config();
        ret.children_index=this.children_index.copy();
        ret.child_dict=this.child_dict.copy();
        ret.dict=this.dict.copy();
        return ret;

    def load(self,xml):
        pass;

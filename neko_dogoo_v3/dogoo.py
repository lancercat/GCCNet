import tensorflow as tf;
from abc import ABC, abstractmethod
from utils.cat_config import cat_config
#factory pattern explained
# https://www.bnmetrics.com/blog/factory-pattern-in-python3-simple-version
#None static dispatcher
#https://stackoverflow.com/questions/9863007/can-we-have-a-static-virtual-functions-if-not-then-why


class dogoo_functor(ABC):
    decay=0.0001;

    def __init__(this):
        this.weight_tree={};
        this.config_tree =cat_config();
        #Comming in V3
        this.submodules=[];

        pass;

    @staticmethod
    def get_variable(shape,initializer,regularizer="default",name="9",**args):
        if(regularizer is "default"):
            regularizer=tf.contrib.layers.l2_regularizer(dogoo_functor.decay);
        return tf.get_variable(name,shape,initializer=initializer,regularizer=regularizer,collections=[tf.GraphKeys.GLOBAL_VARIABLES,tf.GraphKeys.TRAINABLE_VARIABLES],**args);
    @staticmethod
    def variable_scope(name,**args):
        args["reuse"] = args.get("reuse",tf.AUTO_REUSE);
        return tf.variable_scope(name,**args);

    @staticmethod
    def ____not_impl_bug():
        print("err,not_impl");
        exit(9);
    def share_weight_tree(this):
        return this.weight_tree;

    def not_impl(this):
        print(this.get_variable_scope_name());
        print(type(this));
        this.____not_impl_bug();

    def get_variable_scope_name(this, modifier=""):
        return type(this).__name__ + modifier



    @abstractmethod
    def get_default_config(_, **args):
        _.not_impl();

    @abstractmethod
    def _make_weight_dict(_, config_tree, inputs, scope):
        _.not_impl();

    def _infer(_, config_tree, weight_tree, inputs):
        return _.init_ret(config_tree,weight_tree).call(inputs,True);


    def get_weight_for_config(_, config_tree, inputs,placement_preference=None):
        if(placement_preference is not None):
            with tf.device(placement_preference):
                with dogoo_functor.variable_scope(_.get_variable_scope_name()) as scope:
                    weights, outputs = _._make_weight_dict(config_tree, inputs.copy(), scope);
            print("testing feature, will be mature in V4");
        else:
            with dogoo_functor.variable_scope(_.get_variable_scope_name()) as scope:
                weights ,outputs= _._make_weight_dict(config_tree,inputs.copy(),scope);
            # magic string 0xca39_scope --- the value is not a tf variable at all;

        weights["0xca39_scope"]=scope;

        return weights,outputs;

    def inflate_trainable_variables(this):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=this.scope);

    def inflate_model_variables(this):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=this.scope);


    def inflate_global_variables(this):
        return tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope=this.scope);

    def get_saver(this,**args):
        tf.train.Saver(this.inflate_global_variables(),**args);

    def get_saver_from_weight(_,weight_tree,**args):
        scope=weight_tree["0xca39_scope"];
        trainable_weights=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name);
        return  tf.train.Saver(trainable_weights,**args);


    def _init_call_back(this):
        this._init_call_back();

    def init(this,config_tree,weight_tree):
        # config_tree.merge_default(this.get_default_config());
        this.weight_tree=weight_tree;
        this.config_tree=config_tree;
        if("0xca39_scope" in weight_tree):
            this.scope=weight_tree["0xca39_scope"];
        this._init_call_back();

    def init_ret(this, config_tree, weight_tree):
        this.init(config_tree,weight_tree);
        return this;

    def mount_to_parent(this, parent,key):
        this.init(parent.config_tree.get_child(key), parent.weight_tree[key]);
        return this;

    @abstractmethod
    def call(this, inputs,is_training):
        this.not_impl();

class dogoo_backboned_functor(dogoo_functor):

    def get_weight_for_config(_, config_tree, inputs, placement_preference=None):
        print("backboned project_tf_family can't be built alone");
        _.____not_impl_bug();

    def get_default_config(_, **args):
        print("backboned project_tf_family can't get config without all it's backbone");
        _.____not_impl_bug();

    def _make_weight_dict(_,__,___,____):
        print("backboned project_tf_family can't get weight without all it's backbone");
        _.____not_impl_bug();

    @abstractmethod
    def _make_weight_dict_with_backbone(_, config_tree, backbone_weight_tree, inputs, scope):
        _.not_impl();

    @abstractmethod
    def _alter_backbone_config(_, backbone_cfg, **args):
        _.not_impl();

    @abstractmethod
    def _blend_with_other_default_configs(_, altered_backbone_cfg, **args):
        _.not_impl();

    def fill_default_config_with_backbone(_, backbone_cfg, **args):
        altered_backbone_config=_._alter_backbone_config(backbone_cfg.copy(), **args);
        config = _._blend_with_other_default_configs( altered_backbone_config,**args);
        return config;


    def fill_other_weight_for_config(_, config_tree, backbone_weights, inputs, placement_preference=None):
        if (placement_preference is not None):
            with tf.device(placement_preference):
                with dogoo_functor.variable_scope(_.get_variable_scope_name()) as scope:
                    weights, outputs = _._make_weight_dict_with_backbone(config_tree, backbone_weights, inputs.copy(),
                                                                         scope);
                    weights["0xca39_scope"] = scope;
                    print("testing feature, will be mature in V4");
        else:
            with dogoo_functor.variable_scope(_.get_variable_scope_name()) as scope:
                weights, outputs = _._make_weight_dict_with_backbone(config_tree,backbone_weights, inputs.copy(), scope);
                weights["0xca39_scope"]=scope;
        return weights, outputs;




# from  project_tf_family.neko_dogoo_v3.dogoo import dogoo_functor;
# from project_tf_family.neko_dogoo_v3.cat_config import cat_config;

# class xxx_functor(dogoo_functor):
#
#     def __init__(this):
#         super().__init__();
#     def _init_call_back(this):
#         pass;
#     def get_default_config(_,**args):
#         config=cat_config();
#         return config;
#
#     def _make_weight_dict(_, config_tree, inputs, scope):
#
#
#     def call(this, inputs,is_training):


# from  project_tf_family.neko_dogoo_v3.dogoo import dogoo_functor;
# from  project_tf_family.neko_dogoo_v3.dogoo import dogoo_backboned_functor;
# from project_tf_family.neko_dogoo_v3.cat_config import cat_config;

# class xxx_functor(dogoo_backboned_functor):
#
#
#     def __init__(this):
#         super().__init__();
#
#     def _init_call_back(this):
#         pass;
#
#     def _alter_backbone_config(_, backbone_cfg, **args):
#         return backbone_cfg;
#
#     def _blend_with_other_default_configs(_, altered_backbone_cfg, **args):
#         _.not_impl();
#     def _make_weight_dict_with_backbone(_, config_tree,weight_tree, inputs, scope):
#         return other_weights,outputs;
#
#     def call(this, inputs,is_training):
#         pass;

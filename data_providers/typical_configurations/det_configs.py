class kot_det_baseline_data_feeder_config:
    FLAG="";
    def configs(this):
        cfg={};
        cfg["mean"]=this.MEAN;
        cfg["var"]=this.VAR
        cfg["sizes"]=this.SIZES;
        cfg["batch_sizes"]=this.BATCH_SIZES;
        cfg["rotate_ranges_max"]=this.RR_MAXS;
        cfg["rotate_ranges_min"] = this.RR_MINS;
        cfg["rotate_ranges_freq"] = this.RR_FREQS;

        return cfg;


    MEAN = [127, 127, 127];
    VAR = [127, 127, 127];
    BATCH_SIZES = [4];
    SIZES = [640];

    RR_MINS=[];
    RR_MAXS=[];
    RR_FREQS=[];


class kot_det_fots_feeder_config(kot_det_baseline_data_feeder_config):
    FLAG="";
    MEAN = [127, 127, 127];
    VAR = [127, 127, 127];
    BATCH_SIZES = [4];
    SIZES = [640];
    RR_MINS=[-10];
    RR_MAXS=[10];
    RR_FREQS=[1];

class kot_det_baseline_moom_feeder_config(kot_det_baseline_data_feeder_config):
    FLAG="";
    MEAN=[123.68, 116.78, 103.94];
    VAR = [1.,1.,1.];
    SIZES = [512,640,1280];
    BATCH_SIZES = [6,4,1];
    RR_MINS=[-10];
    RR_MAXS=[10];
    RR_FREQS=[1];

class kot_det_baseline_torch_feeder_config(kot_det_baseline_data_feeder_config):
    FLAG="";
    MEAN=[0.485*255, 0.456*255, 0.406*255];
    VAR = [0.229*255.,0.224*255,0.225*255];
    SIZES = [512,640,1280];
    BATCH_SIZES = [6,4,1];
    RR_MINS=[-10];
    RR_MAXS=[10];
    RR_FREQS=[1];
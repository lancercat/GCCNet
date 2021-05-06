
import os;
from data_providers.neko_cerberus_data_loader import neko_cerberus_generator;
from utils.notifier import nekoexpnotification;

class abstract_loadout:
    DATAFEEDER = None;
    DATAFEEDERCFG=None;
    MODEL_FN=None;
    FAMILY_NAME=None;
    DECAY_EACH=10000;
    WORKER_CNT=9;
    DECAY_RATIO=0.94;
    BASE_LR=1e-4;
    GCT=-1;
    FLAG=None;
    PRET=None;

    ################Before runtime;
    def train_setup_callback(this, use_gpu):
        pass;


    def setupmodel(this, model_args):
        this.model=None;
        pass;

    def set_up_saver(this):
        pass;

    def train_setup(this, dataset_cfg, use_gpu=1, total_iter=410000, log_each=20, save_each=10000, ring_each=99990,
                    fpath=None, tpath=None, dssname=None, moving_average_decay=0.997,model_args={}):

        this.dataset_cfg = dataset_cfg;
        this.total_iter = total_iter;
        this.log_each = log_each;
        this.fpath = this.FAMILY_NAME;
        this.tpath = this.FAMILY_NAME;
        if dssname is not None:
            print(this.FAMILY_NAME);
            print(dssname);

            this.fpath = os.path.join(this.fpath, dssname);
            this.tpath = os.path.join(this.tpath, dssname);

        if (fpath is not None):
            this.fpath = fpath;
        if (tpath is not None):
            this.tpath = tpath;

        this.save_each = save_each;
        this.ring_each = ring_each;
        this.PRFX = this.MODEL_FN.PRFX;
        #        this.FLAG = this.MODEL_FN.PRFX + "_" + this.DATAFEEDERCFG.FLAG;

        this.notifier = nekoexpnotification();
        this.admin_mail_address = "lasercat_work@gmx.com";
        this.moving_average_decay = moving_average_decay;
        this.train_setup_callback(use_gpu);

        this.setupmodel(model_args);
        this.set_up_saver();


    #################Before loop
    def reset_feeder(this, feeder):
        if (feeder is not None):
            feeder.stop();
            del feeder;
        cfg = this.DATAFEEDER.get_default_config(this.dataset_cfg,  **this.DATAFEEDERCFG().configs());
        feeder_cfg = neko_cerberus_generator.get_default_config(num_workers=this.WORKER_CNT);
        feeder = neko_cerberus_generator(feeder_cfg);
        feeder.add_task(this.DATAFEEDER, cfg);
        feeder.start();
        return feeder;

    def setup_feeder(this):
        this.feeder=None;
        this.feeder=this.reset_feeder(this.feeder);

    def set_up_optimizer(this, restore_from):
        pass;

    def load_pretrained(this,tpretrained):
        pass;

    def restore(this,restore_from):
        pass;
    ############Within loop

    def save_model(this, iter, loss):
        pass;

    def step(this,iter_id):
        return 0;

    def train(this, restore_from,tpretrained=None):

        this.setup_feeder();
        this.set_up_optimizer(restore_from);
        this.load_pretrained(tpretrained);
        this.restore(restore_from);

        for i in range(restore_from,this.total_iter):
            loss=this.step(i);
            if(i%this.save_each==0):
                this.save_model(i,loss)
                if(i%this.ring_each==0):
                    try:
                        this.notifier.send_msg(this.admin_mail_address,this.PRFX+"reaches"+str(i)+"iters");
                    except :
                        print("failed notifing....");
            pass;
        try:
            this.notifier.send_msg(this.admin_mail_address, this.PRFX + "completed");
        except:
            print("failed notifing....");




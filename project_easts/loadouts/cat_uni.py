from builtins import set

import numpy as np;
from project_easts.loadouts.baseline import otfLoadout_baseline,cat_baseline_evaluator;
from project_easts.easts.cat_uni import cat_uni_kai;
from data_providers.typical_configurations.det_configs import kot_det_baseline_moom_feeder_config,kot_det_baseline_data_feeder_config;


class cat_uni_kai_oom_evaluator(cat_baseline_evaluator):
    MODEL_FN=cat_uni_kai;
    FLAG="uni_kai_oom";
    FEEDER_CFG = kot_det_baseline_moom_feeder_config;

class loadout_uni_kai_oom(otfLoadout_baseline):
    DATAFEEDERCFG=kot_det_baseline_moom_feeder_config
    MODEL_FN = cat_uni_kai;
    FLAG="uniabc2aoom";

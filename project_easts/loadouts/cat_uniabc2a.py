from project_easts.loadouts.cat_abc2a import cat_abc2aoom_evaluator,loadout_abc2aoom;
from project_easts.easts.cat_uni_abc2a import cat_uni_abc2a_kai;
from data_providers.typical_configurations.det_configs import kot_det_baseline_data_feeder_config;

class cat_uniabc2aoom_evaluator(cat_abc2aoom_evaluator):
    MODEL_FN = cat_uni_abc2a_kai;
    FLAG="uniabc2aoom";


class loadout_uniabc2aoom(loadout_abc2aoom):
    MODEL_FN = cat_uni_abc2a_kai;
    FLAG="uniabc2aoom";

class cat_uniabc2aoom_evaluator_res50(cat_abc2aoom_evaluator):
    MODEL_FN = cat_uni_abc2a_kai;
    BBONE = "resnet_v1_50";
    FLAG="uniabc2aoomres50";


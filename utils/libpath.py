import os.path as osp;
import getpass
USR=getpass.getuser();
class pathcfg:
    DATA_ROOT=osp.join("/home",USR,"pubdata");
    USR_ROOT = osp.join("/home", USR, "cat");


    root=osp.join(USR_ROOT,"project_tf_family");
    font_path=osp.join(root,"fonts","OthutomeFont_Ver2.ttf");

    generic_data_root = osp.join(USR_ROOT,"project_tf_generic_data/");
    logs_root = osp.join(USR_ROOT,"project_tf_generic_data");

    pyt_generic_data_root = osp.join(USR_ROOT, "scarlet_generic_data/");

    project_v_root=osp.join(USR_ROOT,"project_tf_family/project_v");
    project_v_data_root=osp.join(USR_ROOT,"project_v_data");

    jerry_mmdet_data_root=osp.join(USR_ROOT,"jerry_mmdet_data");

    tencent_trad_dict=""#""/home/lasercat/cat/project_tf_family/rec_eval/STR_online_final_10514_traditional.txt";
    tencent_simp_dict=""#""/home/lasercat/cat/project_tf_family/rec_eval/STR_online_final_10514_simple.txt";
    tencent_mlt_dict=osp.join(USR_ROOT,"project_tf_family/rec_eval/STR_online_final_15101.txt");


    project_uniabc_data_root=osp.join(USR_ROOT,"project_uniabc_data");
    project_plink_data_root=osp.join(USR_ROOT,"project_plink_data");
    project_rnet_data_root=osp.join(USR_ROOT,"project_rnet_data");
    project_rects_data_root=osp.join(USR_ROOT,"project_rects_data");
    project_rects_east_data_root=osp.join(USR_ROOT,"project_rects_data/EAST");
    project_rects_plink_data_root = osp.join(USR_ROOT,"project_rects_data/plink");

    project_uniabc_root = osp.join(USR_ROOT,"project_tf_family/project_v");
    project_ch_data_root=osp.join(USR_ROOT,"project_ch_data/");

    project_anna_root = osp.join(root,"project_anna");
    project_anna_gt_root = osp.join(project_anna_root,"gts");
    project_anna_data_root =osp.join(USR_ROOT,"project_anna_data/");

    lsvt_train_dataset=osp.join(DATA_ROOT,"datasets/cat/lsvt");
    i15m7_test_dataset = osp.join(DATA_ROOT, "datasets/testing_sets/ch4_test_imagesmb7");
    i15m5_test_dataset = osp.join(DATA_ROOT, "datasets/testing_sets/ch4_test_imagesmb5");
    i15m3_test_dataset = osp.join(DATA_ROOT, "datasets/testing_sets/ch4_test_imagesmb3");
    i15m1_test_dataset = osp.join(DATA_ROOT, "datasets/testing_sets/ch4_test_imagesmb1");
    i15msnp_test_dataset = osp.join(DATA_ROOT, "datasets/testing_sets/ch4_test_imagessnp");

    i15_val_dataset = osp.join(DATA_ROOT, "datasets/testing_sets/ch4_training_images");
    i15_test_dataset=osp.join(DATA_ROOT,"datasets/testing_sets/ch4_test_images");
    mlt_val_dataset=osp.join(DATA_ROOT,"datasets/testing_sets/mlt_val");
    mlt_eval_dataset=osp.join(DATA_ROOT,"datasets/testing_sets/mlt_eval");
    lsvt_val_dataset = osp.join(DATA_ROOT,"datasets/testing_sets/lsvt_val");


    msra_test_dataset=osp.join(DATA_ROOT,"datasets/testing_sets/td500-test");
    sv1k_test_dataset=osp.join(DATA_ROOT,"datasets/testing_sets/sv1k-test");
    rects_val_dataset=osp.join(DATA_ROOT,"datasets/testing_sets/rects_val_mod5");


    coco_test_dataset=osp.join(DATA_ROOT,"datasets/cat/coco/test/");
    training_dataset_root=osp.join(DATA_ROOT,"datasets/cat");

    pkey = osp.join("/home", USR, "hakurei","reimu");




class alien_pathcfg:
    def __init__(this,USR):
        this.root=osp.join("/home",USR,"data/cat/project_tf_family");

        this.generic_data_root = osp.join("/home", USR, "cat/project_tf_generic_data/");

        this.project_v_root=osp.join("/home",USR,"data/cat/project_tf_family/project_v");
        this.project_v_data_root=osp.join("/home",USR,"cat/project_v_data");

        this.project_uniabc_data_root=osp.join("/home",USR,"cat/project_uniabc_data");
        this.project_plink_data_root=osp.join("/home",USR,"cat/project_plink_data");
        this.project_rnet_data_root=osp.join("/home",USR,"cat/project_rnet_data");
        this.project_rects_data_root=osp.join("/home",USR,"cat/project_rects_data");
        this.project_rects_east_data_root=osp.join("/home",USR,"cat/project_rects_data/EAST");
        this.project_rects_plink_data_root = osp.join("/home", USR, "cat/project_rects_data/plink");

        this.project_uniabc_root = osp.join("/home",USR,"data/cat/project_tf_family/project_v");
        this.project_ch_data_root=osp.join("/home",USR,"cat/project_ch_data/");

        this.project_anna_root = osp.join(this.root,"project_anna");
        this.project_anna_gt_root = osp.join(this.project_anna_root,"gts");
        this.project_anna_data_root =osp.join("/home",USR,"cat/project_anna_data/");

        this.i15_test_dataset=osp.join("/home", USR, "pubdata/datasets/testing_sets/ch4_test_images");
        this.i15m7_test_dataset = osp.join(DATA_ROOT, "datasets/testing_sets/ch4_test_imagesmb7");
        this.i15m5_test_dataset = osp.join(DATA_ROOT, "datasets/testing_sets/ch4_test_imagesmb5");
        this.i15m3_test_dataset = osp.join(DATA_ROOT, "datasets/testing_sets/ch4_test_imagesmb3");
        this.i15m1_test_dataset = osp.join(DATA_ROOT, "datasets/testing_sets/ch4_test_imagesmb1");
        this.i15msnp_test_dataset = osp.join(DATA_ROOT, "datasets/testing_sets/ch4_test_imagessnp");

        this.mlt_val_dataset=osp.join("/home",USR,"pubdata/datasets/testing_sets/mlt_val");
        this.mlt_eval_dataset=osp.join("/home",USR,"pubdata/datasets/testing_sets/mlt_eval");
        this.msra_test_dataset=osp.join("/home",USR,"pubdata/datasets/testing_sets/td500-test");
        this.sv1k_test_dataset=osp.join("/home",USR,"pubdata/datasets/testing_sets/sv1k-test");
        this.rects_val_dataset = osp.join("/home", USR, "pubdata/datasets/testing_sets/rects_val_mod5");

        this.coco_test_dataset=osp.join("/home",USR,"pubdata/datasets/cat/coco/test/");
        this.training_dataset_root=osp.join("/home",USR,"pubdata/datasets/cat");

        this.pkey = osp.join("/home", USR, "hakurei","reimu");



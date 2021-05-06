
from project_easts.easts.baseline import cat_welf_east_baseline;
from project_easts.cat_uni_loss import cat_welf_east_uni_ohem_loss_functor_kai,cat_welf_east_uni_ohem_loss_functor_kai_ws,cat_welf_east_uni_ohem_loss_functor_kaif,\
    cat_welf_east_uni_ohem_loss_functor_kaiII,cat_welf_east_uni_ohem_loss_functor_kaiIIf;


class cat_uni_kai(cat_welf_east_baseline):
    PRFX = "uni_kai";
    LOSS_FN = cat_welf_east_uni_ohem_loss_functor_kai;
from project_easts.easts.cat_abc2a import cat_abc2a
from project_easts.cat_uni_loss import cat_welf_east_uni_ohem_loss_functor_kaif;


class cat_uni_abc2a_kai(cat_abc2a):
    PRFX = "uniabc2a";
    LOSS_FN = cat_welf_east_uni_ohem_loss_functor_kaif;

import numpy as np;
def kot_V_generic(raw_entry,required,**kwargs):
    required[0].append(np.array([raw_entry.has_gts]).astype(np.float32))


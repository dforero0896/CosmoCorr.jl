import numpy as np
import sys
sys.path.append("/home/astro/dforero/codes/pypowspec/powspec/")
from pypowspec import compute_auto_box, compute_cross_box


data_cat_fn = "/home/astro/dforero/codes/BAOrec/data/CATALPTCICz0.466G960S1010008301_zspace.dat.npy"
data = np.load(data_cat_fn)
data += 2500
data %= 2500
w = np.ones(data.shape[0], dtype = data.dtype)
pk = compute_auto_box(data[:,0], data[:,1], data[:,2], w, 
                      powspec_conf_file = "test/powspec_auto.conf",
                      output_file = "test/box_auto_test.powspec")
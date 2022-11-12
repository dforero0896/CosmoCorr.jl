import numpy as np
from Corrfunc.theory.DD import DD

void_fn = "../data/VOID_CATALPTCICz0.466G960S1005638091.npy"
void_data = np.load(void_fn)
box_size = 1000.
mask = (void_data[:,:3] < box_size).all(axis=1) & (void_data[:,3] > 16.) & (void_data[:,3] < 50.)
void_data = void_data[mask]


results = DD(1, 16, np.arange(1e-4, 205., 5.), void_data[:,0], void_data[:,1], void_data[:,2], weights1=np.ones_like(void_data[:,1]),
              weight_type='pair_product', output_ravg=True,
              boxsize=box_size, periodic=True)

print(results['npairs'])
np.savetxt("test_cf_tpcf.dat", np.c_[results['ravg'], results['npairs'], results['npairs'] / (void_data.shape[0]**2)])
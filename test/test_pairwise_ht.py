import numpy as np
import pandas as pd
import halotools.mock_observables
import proplot as pplt
import time


box_size = 2500.
def mask_box(data, box_size):
    mask = (data[:,:3] < box_size).all(axis=1)
    return data[mask]

real_void = np.load("../data/VOID_CATALPTCICz0.466G960S1005638091.npy").astype(np.float32)
real_gal = pd.read_csv("../data/CATALPTCICz0.466G960S1005638091.dat", delim_whitespace=True, names = ['x', 'y', 'z', 'vx', 'vy', 'vz'], usecols=(0,1,2,3,4,5)).values.astype(np.float32)

_t = time.time()
void_mask = real_void[:,3] > 16.
#void_mask = real_void[:,-2] < (real_gal.shape[0] / box_size**3)
real_void = mask_box(real_void[void_mask], box_size)
real_gal = mask_box(real_gal, box_size)
print(f"Masking in {time.time() - _t}", flush=True)


bin_edges = np.arange(1e-5, 121., 1.)

# Since sample1 is voids, according to https://halotools.readthedocs.io/en/latest/api/halotools.mock_observables.mean_radial_velocity_vs_r.html
# this computes v1.r12 - v2.r12 so to obtain v2.r12 we should multiply by -1 but the shape seems odd
# the odd shape may be because there could be another minus from r12 = |r1-r2| = |rv - rg| (like in https://arxiv.org/pdf/1712.07575.pdf)
print(f"Computing pairwise unnormalized", flush=True)
_t =  time.time()
v_12 = halotools.mock_observables.mean_radial_velocity_vs_r(sample1 = real_void[:,:3],
                                                            velocities1 = np.broadcast_to(np.array([0.]), real_void[:,:3].shape),
                                                            rbins_absolute = bin_edges,
                                                            sample2 = real_gal[:,:3],
                                                            velocities2 = real_gal[:,3:7],
                                                            period = box_size,
                                                            num_threads = 64)
print(f"Histogram in {time.time() - _t}", flush=True)
bin_centers = bin_edges[:-1] + 0.5 * np.diff(bin_edges)
np.savetxt("test_ht_abs.dat", np.c_[bin_edges[:-1], bin_edges[1:], bin_centers, v_12])


bin_edges = np.arange(1e-5, 5., 1e-1)
print(f"Computing pairwise normalized", flush=True)
_t =  time.time()
v_12 = halotools.mock_observables.mean_radial_velocity_vs_r(sample1 = real_void[:,:3],
                                                            velocities1 = np.broadcast_to(np.array([0.]), real_void[:,:3].shape),
                                                            normalize_rbins_by=real_void[:,3],
                                                            rbins_normalized = bin_edges,
                                                            sample2 = real_gal[:,:3],
                                                            velocities2 = real_gal[:,3:7],
                                                            period = box_size,
                                                            num_threads = 64)
bin_centers = bin_edges[:-1] + 0.5 * np.diff(bin_edges)
print(f"Histogram in {time.time() - _t}", flush=True)
bin_centers = bin_edges[:-1] + 0.5 * np.diff(bin_edges)
np.savetxt("test_ht_norm.dat", np.c_[bin_edges[:-1], bin_edges[1:], bin_centers, v_12])


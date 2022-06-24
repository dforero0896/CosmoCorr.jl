import numpy as np
import matplotlib
matplotlib.use("Agg")
import proplot as pplt





fig, ax = pplt.subplots(nrows=1, ncols=2, share=0)
s, vnorm = np.loadtxt("test_cc_abs.dat", usecols=(2, 5), delimiter=',', skiprows=1, unpack=True)
ax[0].plot(s, vnorm, label = "CosmoCorr")
s, vnorm = np.loadtxt("test_ht_abs.dat", usecols=(2, 3), unpack=True)
ax[0].plot(s, vnorm, label = "HaloTools", ls='--')

s, vnorm = np.loadtxt("test_cc_norm.dat", usecols=(2, 5), delimiter=',', skiprows=1, unpack=True)
ax[1].plot(s, vnorm, label = "CosmoCorr")
s, vnorm = np.loadtxt("test_ht_norm.dat", usecols=(2, 3), unpack=True)
ax[1].plot(s, vnorm, label = "HaloTools", ls='--')
ax[0].format(xlabel='$s$ [Mpc/h]', ylabel = r'$\bar{v}_{12} [km/s]$')
ax[1].format(xlabel='$s$ / $R_v$', ylabel = r'$\bar{v}_{12} [km/s]$')
ax.legend(loc='top')
fig.savefig("../plots/test_pairwise.png", dpi=300)


fig, ax = pplt.subplots(nrows=1, ncols=2, share=0)
s, dd, dd_norm, xi, rr_norm = np.loadtxt("test_cc_tpcf.dat", usecols=(2, 4, 5, 6, 7), delimiter=',', skiprows=1, unpack=True)
print(xi)
print(dd)
print(s)
ax[0].plot(s, s**2*(xi))
s, dd, dd_norm= np.loadtxt("test_cc_tpcf.dat", usecols=(2, 4, 5), delimiter=',', skiprows=1, unpack=True)
ax[0].plot(s, s**2*(dd_norm / rr_norm - 1.), ls='--')
fig.savefig("../plots/test_pair_counters.png", dpi=300)


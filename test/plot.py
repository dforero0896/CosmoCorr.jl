import numpy as np
import matplotlib
matplotlib.use("Agg")
import proplot as pplt





fig, ax = pplt.subplots(nrows=1, ncols=1, share=0)
s, vnorm = np.loadtxt("test/threaded_sub.dat", usecols=(2, 5), delimiter=',', skiprows=1, unpack=True)
ax[0].plot(s, vnorm)

s, vnorm = np.loadtxt("test/threaded_cl.dat", usecols=(2, 5), delimiter=',', skiprows=1, unpack=True)
ax[0].plot(s, vnorm)
fig.savefig("plots/test.png", dpi=300)


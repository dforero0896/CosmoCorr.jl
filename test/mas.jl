import Pkg, Revise
Pkg.add(path="/home/astro/dforero/codes/CosmoCorr")
Pkg.resolve()


#import Pkg; Pkg.add("Plots")
using CosmoCorr
using Random
using Plots


box_size = 1000.
n_bins = 256
data_cat = box_size * rand(Float32, (100,3))
rho = zeros(Float32, n_bins, n_bins)

CosmoCorr.cic(data_cat[:,1], data_cat[:,2], data_cat[:,3], [1000., 1000., 1000.], [0.,0.,0.], rho)

#image = rand(Float32, (100,100))
#savefig(heatmap(image), "plots/mas.png")
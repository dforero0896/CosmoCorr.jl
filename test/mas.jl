using CosmoCorr
using Random

data_cat = Matrix{Float32}(undef, 1e6, 3)
fill!(data_cat, 0.)
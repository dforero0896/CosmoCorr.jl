using Revise
using CosmoCorr
using NPZ
using Plots
using CSV
using StaticArrays
using Statistics
using FFTW
using DataFrames
FFTW.set_num_threads(64)

data_cat_fn = "/home/astro/dforero/codes/BAOrec/data/CATALPTCICz0.466G960S1010008301_zspace.dat.npy"
data_cat_pos = npzread(data_cat_fn)
data_cat_pos = [data_cat_pos[:,i] for i in 1:3]
data_cat_w = zero(data_cat_pos[1]) .+ 1

box_size = @SVector [2500f0 for _ in 1:3]
box_min = @SVector [0f0 for _ in 1:3]
const grid_size = (512, 512, 512)

ρ = zeros(eltype(data_cat_pos[1]), grid_size...);
cic!(ρ, data_cat_pos..., data_cat_w, box_size, box_min; wrap = true)

p = heatmap(dropdims(mean(ρ, dims=1), dims=1))

savefig(p, "plots/simulation.jl")




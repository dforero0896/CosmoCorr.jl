using CosmoCorr
using Test
using Random
using Plots
using Statistics
using BenchmarkTools
using CSV
using DataFrames

#include("test_pairwise.jl")

#=
HALOS="/hpcstorage/zhaoc/PATCHY_BOX/pre-recon/halo/BDM_Apk/CATALPTCICz0.562G960S1010008301.dat"
data_cat = Matrix(DataFrame(CSV.File(HALOS, delim=" ")))
println(size(data_cat)

@testset "CosmoCorr.jl" begin
    # Write your tests here.
    box_size = 2500.
    n_bins = 256
    rho = zeros(Float32, n_bins, n_bins, n_bins)
    @test CosmoCorr.cic(data_cat[:,1], data_cat[:,2], data_cat[:,3], ones(Float32, size(data_cat)[1]), [box_size, box_size, box_size], [0.,0.,0.], rho) == 0

 
end

rho = zeros(Float32, n_bins, n_bins, n_bins)
@btime CosmoCorr.cic(data_cat[:,1], data_cat[:,2], data_cat[:,3], ones(Float32, size(data_cat)[1]), [box_size, box_size, box_size], [0.,0.,0.], rho) == 0
toplot = mean(rho, dims=3)[:,:,1]
println(size(toplot))
p = heatmap(convert(Array{Float64}, toplot))

savefig(p, "../plots/mas.png")
=#
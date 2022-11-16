using Revise
using CosmoCorr
using NPZ
using Plots
using CSV
using StaticArrays
using Statistics
using FFTW
using DataFrames
using LinearAlgebra
FFTW.set_num_threads(64)
test_pk = DataFrame(CSV.File("test/box_auto_test.powspec", delim=" ", header=[:k, :d1, :d2, :d3, :d4, :P0, :P2, :P4], types = [Float32 for _ in 1:8], ignorerepeated = true, comment = "#"))
p2 = plot(test_pk[!,:k], test_pk[!,:k] .* test_pk[!,:P0])
p3 = plot(test_pk[!,:k], test_pk[!,:k] .* test_pk[!,:P2])
p4 = plot(test_pk[!,:k], test_pk[!,:k] .* test_pk[!,:P4])
p = plot(p2, p3, p4, layout = (1,3))
savefig(p, "plots/simulation.png")
#exit()
data_cat_fn = "/home/astro/dforero/codes/BAOrec/data/CATALPTCICz0.466G960S1010008301_zspace.dat.npy"
data_cat_pos = npzread(data_cat_fn)
data_cat_pos = [data_cat_pos[:,i] for i in 1:3]
data_cat_w = zero(data_cat_pos[1]) .+ 1

box_size = @SVector [2500f0 for _ in 1:3]
box_min = @SVector [0f0 for _ in 1:3]
const grid_size = Tuple(256 for i_ in 1:3)

mesh = CosmoCorr.Mesh(data_cat_pos, data_cat_w, grid_size, box_size, box_min, 2)
p = heatmap(dropdims(mean(mesh.field_r, dims=1), dims=1), aspect_ratio = :equal)
savefig(p, "plots/simulation.png")
mul!(mesh.field_k, mesh.fft_plan, mesh.field_r)
k⃗ = CosmoCorr.k_vec(mesh.field_r, box_size) 
const k_edges = range(0f0, maximum(π .* grid_size ./ box_size), step = 5f-3)
const los = [0f0, 0f0, 1f0]
power, modes = CosmoCorr.count_modes(mesh, [0,2,4], k_edges, true, k⃗, los)
k = 0.5 .* (k_edges[2:end] .+ k_edges[1:end-1])
p2 = plot!(p2, k,  k.^1 .* real.(power[1,:]))#, xscale = :log10)
p3 = plot!(p3, k,  k.^1 .* real.(power[2,:]))#, xscale = :log10)
p4 = plot!(p4, k,  k.^1 .* real.(power[3,:]))#, xscale = :log10)
#exit()




p = plot(p, p2, p3, p4, layout = (2,2))

savefig(p, "plots/simulation.png")

@show mesh
@show modes
@show real.(power[1,:])
@show real.(power[2,:])
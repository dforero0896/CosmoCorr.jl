using Plots
using CSV
using DataFrames
using NearestNeighbors
using CellListMap
using NPZ
using StaticArrays
using LinearAlgebra
using CosmoCorr

gal_fn = "../data/CATALPTCICz0.466G960S1005638091.dat"
void_fn = "../data/VOID_CATALPTCICz0.466G960S1005638091.npy"

mutable struct VoidCat
    pos::Vector{SVector{3,Float32}}
    rad::Vector{Float32}
    ndens::Vector{Float32}
end
mutable struct GalCat
    pos::Vector{SVector{3,Float32}}
    vel::Vector{SVector{3,Float32}}
end

function mask_void_cat!(void_cat::VoidCat, min::Real, max::Real, fieldname::Symbol)
    tomask = getfield(void_cat, fieldname)
    mask = (tomask .> min) .& (tomask .< max)    
    void_cat.pos = void_cat.pos[mask]
    void_cat.rad = void_cat.rad[mask]   
    void_cat.ndens = void_cat.ndens[mask]   
end
function mask_data_box!(data, pos, box_size)
    mask = [all(v .< box_size) for v in pos]
    data = data[mask]
end

box_size = SVector{3, Float64}(500., 500., 500.)
gal_data = Array{Float32,2}(Array{Float32, 2}(CSV.read(gal_fn, DataFrame, header=0, types=Float32))')
gal_cat = GalCat(Vector{SVector{3, Float32}}([SVector{3, Float32}(vector) for vector in eachslice(gal_data[1:3,:], dims=2)]),
                 Vector{SVector{3, Float32}}([SVector{3, Float32}(vector) for vector in eachslice(gal_data[4:6,:], dims=2)]))
void_data = Array{Float32,2}(npzread(void_fn)')
void_cat = VoidCat(Vector{SVector{3, Float32}}([SVector{3, Float32}(vector) for vector in eachslice(void_data[1:3,:], dims=2)]),
                   Vector{Float32}(void_data[4,:]),
                   Vector{Float32}(void_data[end-1,:]))

mask_void_cat!(void_cat, 16, 50, :rad)
#mask_void_cat!(void_cat, 0., 3.9e-4, :ndens)

mask = [all(v .< box_size) for v in void_cat.pos]
void_cat.rad = void_cat.rad[mask]
void_cat.ndens = void_cat.ndens[mask]
void_cat.pos = void_cat.pos[mask]
mask = [all(v .< box_size) for v in gal_cat.pos]
gal_cat.vel = gal_cat.vel[mask]
gal_cat.pos = gal_cat.pos[mask]


bin_edges = collect(1e-8:5.:205)
println("Computing box pair counts")
n_pairs, weighted_pairs = @time CosmoCorr.box_paircount_cellist(void_cat.pos,
                                                    [1. for _ in 1:size(void_cat.pos)[1]],
                                                    void_cat.pos,
                                                    [1. for _ in 1:size(void_cat.pos)[1]],
                                                    bin_edges,
                                                    box_size,
                                                    )
#norm = length(void_cat.pos) * length(gal_cat.pos)
norm = length(void_cat.pos)^2
@show norm

rr_counts = analytic_rr_1d(bin_edges, box_size)
@show rr_counts
@show weighted_pairs ./ norm
@show (weighted_pairs[1:end-1] ./ norm ./ rr_counts) .- 1
bin_centers = 0.5 * (bin_edges[1:end-1] + bin_edges[2:end])
CSV.write("test_cc_tpcf.dat", (smin = bin_edges[1:end-1], smax = bin_edges[2:end], scen = bin_centers, n_pairs=n_pairs, wpairs = weighted_pairs, norm_wpairs = weighted_pairs ./ norm, tpcf = weighted_pairs[1:end-1] ./ norm ./ rr_counts .- 1.))

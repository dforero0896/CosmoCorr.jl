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

box_size = SVector{3, Float64}(2500., 2500., 2500.)
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


bin_edges = collect(1e-5:1.:121)
distance_cutoff = 120.
println("Computing pairwise unnormalized")
n_pairs, v_r = @time CosmoCorr.pairwise_vel_cellist(void_cat.pos,
                                [SVector{3,Float32}(0., 0., 0.) for _ in 1:size(void_cat.pos)[1]], 
                                gal_cat.pos,
                                gal_cat.vel,
                                bin_edges,
                                box_size,
                                distance_cutoff,
                                [1. for _ in 1:size(void_cat.pos)[1]]
                                )

bin_centers = 0.5 * (bin_edges[1:end-1] + bin_edges[2:end])
CSV.write("test_cc_abs.dat", (smin = bin_edges[1:end-1], smax = bin_edges[2:end], scen = bin_centers, n_pairs=n_pairs, vr = v_r, vrnorm = v_r ./ n_pairs))


bin_edges = collect(1e-5:1e-1:5)
distance_cutoff = maximum(bin_edges) * maximum(void_cat.rad)
println("Computing pairwise normalized")
n_pairs, v_r = @time CosmoCorr.pairwise_vel_cellist(void_cat.pos,
                                [SVector{3,Float32}(0., 0., 0.) for _ in 1:size(void_cat.pos)[1]], 
                                gal_cat.pos,
                                gal_cat.vel,
                                bin_edges,
                                box_size,
                                distance_cutoff,
                                void_cat.rad,
                                )

bin_centers = 0.5 * (bin_edges[1:end-1] + bin_edges[2:end])
CSV.write("test_cc_norm.dat", (smin = bin_edges[1:end-1], smax = bin_edges[2:end], scen = bin_centers, n_pairs=n_pairs, vr = v_r, vrnorm = v_r ./ n_pairs))


#=
#bin_edges = collect(1e-5:1:151)
n_pairs, v_r = @time CosmoCorr.pairwise_vel_balltree_loop_threaded(void_cat.pos,
                                [SVector{3,Float32}(0., 0., 0.) for _ in 1:size(void_cat.pos)[1]], 
                                gal_cat.pos,
                                gal_cat.vel,
                                bin_edges,
                                box_size,
                                distance_cutoff,
                                void_cat.rad,
                                #[16. for _ in 1:size(void_cat.pos)[1]]
                                )

bin_centers = 0.5 * (bin_edges[1:end-1] + bin_edges[2:end])
#p = plot(bin_centers, v_r ./ n_pairs)
#savefig(p, "threaded_sub.png")
CSV.write("threaded_sub.dat", (smin = bin_edges[1:end-1], smax = bin_edges[2:end], scen = bin_centers, n_pairs=n_pairs, vr = v_r, vrnorm = v_r ./n_pairs))
=#

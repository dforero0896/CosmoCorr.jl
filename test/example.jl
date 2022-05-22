using Pkg
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("NearestNeighbors")
Pkg.add("CellListMap")
Pkg.add("NPZ")
Pkg.add("StaticArrays")
Pkg.add("LinearAlgebra")

using Plots
using CSV
using DataFrames
using NearestNeighbors
using CellListMap
using NPZ
using StaticArrays
using LinearAlgebra

push!(LOAD_PATH, "/home/astro/dforero/codes/CosmoCorr/")
using CosmoCorr
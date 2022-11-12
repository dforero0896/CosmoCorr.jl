module CosmoCorr

using LoopVectorization
using Parameters
using StaticArrays
using LinearAlgebra
using AbstractFFTs: fftfreq, rfftfreq
using FFTW
using Statistics
using QuadGK
using Interpolations
using CUDA
using KernelAbstractions
using CUDAKernels

# Write your package code here.


include("mas.jl")
include("power_spectrum.jl")
#include("pairwise_velocities.jl")
#include("pair_counters.jl")

end

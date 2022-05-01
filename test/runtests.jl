using CosmoCorr
using Test
using Random

@testset "CosmoCorr.jl" begin
    # Write your tests here.
    box_size = 1000.
    n_bins = 256
    data_cat = box_size * rand(Float32, (100,3))
    rho = zeros(Float32, n_bins, n_bins)
    @test CosmoCorr.cic(data_cat[:,1], data_cat[:,2], data_cat[:,3], [1000., 1000., 1000.], [0.,0.,0.], rho)
end

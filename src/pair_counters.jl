
using NearestNeighbors
using Distances
using NPZ
using StaticArrays
using LinearAlgebra
using CellListMap


export box_paircount_cellist



function analytic_rr_1d(bin_edges::Vector, box_size::SVector{3,<:AbstractFloat})
    return 4 * pi .*  (bin_edges[2:end].^3 .- bin_edges[1:end-1].^3) ./ (box_size[1] * box_size[2] * box_size[3] * 3)
end



function map_box_pc(x, y, i, j, d2, output, wx, wy, bin_edges, box_size)
    #s_vector = x - y #CellListMap takes care of PBC already but a small shift is seen in the final hist.
    #s_vector = separation_vector(x,y, box_size)
    bin_id = searchsortedlast(bin_edges, sqrt(d2))
    if (bin_id > 0) & (bin_id < length(bin_edges))
        output[1][bin_id] += (wx[i] * wy[j])
        output[2][bin_id] += 1
    end
    return output
end


function smu_map_box_pc(x, y, i, j, d2, output, wx, wy, bin_edges, box_size, los)
    
    #s_vector = x - y #CellListMap takes care of PBC already but a small shift is seen in the final hist.
    s_vector = separation_vector(x,y, box_size)
    s_dot_l = sum(s_vector .* los)
    l² = sum(los.^2)
    s² = sum(s_vector.^2)
    s_dot_l² = s_dot_l^2
    μ² = s_dot_l² / (l² * s²)
    μ = sqrt(μ²)
    
    bin_id = searchsortedlast(bin_edges, sqrt(d2))
    if (bin_id > 0) & (bin_id < length(bin_edges))
        output[1][bin_id] += (wx[i] * wy[j])
        output[2][bin_id] += 1
    end
    return output
end

function map_box_diff_pc(x, y, i, j, d2, output, wx, wy, bin_edges, box_size)
    #s_vector = x - y #CellListMap takes care of PBC already but a small shift is seen in the final hist.
    #s_vector = separation_vector(x,y, box_size)
    bin_id = searchsortedlast(bin_edges, sqrt(d2))
    if (bin_id > 0) & (bin_id < length(bin_edges))
        output[1][bin_id] += (wx[i] * wy[j])
        output[2][bin_id] += 1
    end
    return output
end

function reduce_hist(hist,hist_threaded)
    hist[1] .= hist_threaded[1][1]
    hist[2] .= hist_threaded[1][2]
    for i in 2:length(hist_threaded) # see https://m3g.github.io/CellListMap.jl/stable/parallelization/#Number-of-batches
     hist[1] .+= hist_threaded[i][1]
     hist[2] .+= hist_threaded[i][2]
    end
    return hist
  end



function iso_box_paircount_cellist(sample_1::Vector{SVector{3,Float32}},
                                    weight_1::Vector{<:AbstractFloat}, 
                                    sample_2::Vector{SVector{3,Float32}},
                                    weight_2::Vector{<:AbstractFloat},
                                    bin_edges::Vector,
                                    box_size::SVector{3},
                                    )
    
    n_bins = size(bin_edges)
    distance_cutoff = maximum(bin_edges)
    box = Box(box_size, distance_cutoff)
    cl = CellList(sample_1, sample_2, box)
    output = (zeros(n_bins),
            zeros(Int32, n_bins))
    map_pairwise!(
        (x, y, i, j, d2, output) -> map_box_pc(x, y, i, j, d2, output, weight_1, weight_2, bin_edges, box_size),
        output, box, cl,
        reduce = reduce_hist,
        show_progress = true
    )

    return output[2], output[1]
end


function smu_box_paircount_cellist(sample_1::Vector{SVector{3,Float32}},
        weight_1::Vector{<:AbstractFloat}, 
        sample_2::Vector{SVector{3,Float32}},
        weight_2::Vector{<:AbstractFloat},
        bin_edges::Vector,
        box_size::SVector{3},
        )

    n_bins = size(bin_edges)
    distance_cutoff = maximum(bin_edges)
    box = Box(box_size, distance_cutoff)
    cl = CellList(sample_1, sample_2, box)
    output = (zeros(n_bins),
    zeros(Int32, n_bins))
    map_pairwise!(
    (x, y, i, j, d2, output) -> smu_map_box_pc(x, y, i, j, d2, output, weight_1, weight_2, bin_edges, box_size),
    output, box, cl,
    reduce = reduce_hist,
    show_progress = true
    )

    return output[2], output[1]
end


#precompile(box_paircount_cellist, (Vector{SVector{3,Float32}},
#                                    Vector{<:AbstractFloat}, 
#                                    Vector{SVector{3,Float32}},
#                                    Vector{<:AbstractFloat},
#                                    Vector,
#                                    SVector{3},
#                                    ))
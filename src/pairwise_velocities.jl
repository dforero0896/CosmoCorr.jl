
using NearestNeighbors
using Distances
using NPZ
using StaticArrays
using LinearAlgebra
using CellListMap


export pairwise_vel_balltree

function coordinate_separation(a, b, box_size)
    delta = abs(a - b)
    return (delta > 0.5*box_size ? delta - box_size : delta)*sign(a-b)
end

function separation_vector(a, b, box_size)
    return SVector{3,Float32}(ntuple(i -> coordinate_separation(a[i],b[i],box_size[i]), 3))
end

function pairwise_vel_balltree(sample_1::Vector{SVector{3,Float32}},
                               vel_1::Vector{SVector{3,Float32}}, 
                               sample_2::Vector{SVector{3,Float32}},
                               vel_2::Vector{SVector{3,Float32}},
                               bin_edges::Vector,
                               box_size::SVector{3},
                               distance_cutoff::AbstractFloat,
                               distance_normalizer::Vector{<:AbstractFloat},
                               )
                
    sample_tree_1 = BallTree(sample_1, PeriodicEuclidean(box_size), leafsize = 10, reorder=false)
    n_bins = size(bin_edges,1)
    v_r = zeros(size(bin_edges))
    n_pairs = zeros(Int32, size(bin_edges))
    for i in 1:size(sample_2)[1]
        idx = inrange(sample_tree_1, sample_2[i], distance_cutoff,false)
        s_vectors = separation_vector.(sample_tree_1.data[idx], Ref(sample_2[i]), Ref(box_size))
        norms = LinearAlgebra.norm.(s_vectors)
        bin_id = searchsortedfirst.(Ref(bin_edges), norms ./ distance_normalizer[idx])
        if bin_id < length(bin_edges) -1
            v_r[bin_id] .+= (LinearAlgebra.dot.(vel_1[idx] .- Ref(vel_2[i]), s_vectors) ./ norms)
            n_pairs[bin_id] .+= 1
        end
    end
    return n_pairs, v_r
end

function pairwise_vel_balltree_loop(sample_1::Vector{SVector{3,Float32}},
                                    vel_1::Vector{SVector{3,Float32}}, 
                                    sample_2::Vector{SVector{3,Float32}},
                                    vel_2::Vector{SVector{3,Float32}},
                                    bin_edges::Vector,
                                    box_size::SVector{3},
                                    distance_cutoff::AbstractFloat,
                                    distance_normalizer::Vector{<:AbstractFloat},
                                    )

    sample_tree_1 = BallTree(sample_1, PeriodicEuclidean(box_size), leafsize = 10, reorder=false)
    n_bins = size(bin_edges,1)
    v_r = zeros(size(bin_edges))
    n_pairs = zeros(Int32, size(bin_edges))
    for i in 1:size(sample_2)[1]
        idx = inrange(sample_tree_1, sample_2[i], distance_cutoff,false)
        for j in 1:size(idx)[1]
            s_vectors = separation_vector(sample_tree_1.data[idx[j]], sample_2[i], box_size)
            norms = LinearAlgebra.norm(s_vectors)
            bin_id = searchsortedfirst(bin_edges, norms / distance_normalizer[idx[j]]) 
            if bin_id < length(bin_edges) -1
                v_r[bin_id] += (LinearAlgebra.dot(vel_1[idx[j]] - vel_2[i], s_vectors) / norms)
                n_pairs[bin_id] += 1
            end
        end
    end
    return n_pairs, v_r
end


function pairwise_vel_balltree_loop_threaded(sample_1::Vector{SVector{3,Float32}},
                                            vel_1::Vector{SVector{3,Float32}}, 
                                            sample_2::Vector{SVector{3,Float32}},
                                            vel_2::Vector{SVector{3,Float32}},
                                            bin_edges::Vector,
                                            box_size::SVector{3},
                                            distance_cutoff::AbstractFloat,
                                            distance_normalizer::Vector{<:AbstractFloat},
                                            )
    @assert size(sample_1) == size(vel_1)
    @assert size(sample_2) == size(vel_2)
    sample_tree_1 = BallTree(sample_1, PeriodicEuclidean(box_size), leafsize = 10, reorder=false)
    n_bins = size(bin_edges,1)
    v_r = [zeros(size(bin_edges)) for t in 1:Threads.nthreads()]
    n_pairs = [zeros(Int32, size(bin_edges)) for t in 1:Threads.nthreads()]
    Threads.@threads for i in 1:size(sample_2)[1]
        idx = inrange(sample_tree_1, sample_2[i], distance_cutoff,false)
        for j in 1:size(idx)[1]
            s_vectors = separation_vector(sample_tree_1.data[idx[j]], sample_2[i], box_size)
            norms = LinearAlgebra.norm(s_vectors)
            bin_id = searchsortedfirst(bin_edges, norms / distance_normalizer[idx[j]])
            if bin_id < length(bin_edges) -1
                v_r[Threads.threadid()][bin_id] += (LinearAlgebra.dot(vel_1[idx[j]] - vel_2[i], s_vectors) / norms)
                n_pairs[Threads.threadid()][bin_id] += 1
            end
        end
    end
    return sum(n_pairs), sum(v_r)
end


function map_function(x, y, i, j, d2, output, vx, vy, bin_edges, box_size, distance_normalizer)
    s_vector = x - y #CellListMap takes care of PBC already but a small shift is seen in the final hist.
    #s_vector = separation_vector(x,y, box_size)
    norm = sqrt(d2)
    bin_id = searchsortedfirst(bin_edges, norm / distance_normalizer[i]) 
    if bin_id < length(bin_edges) -1
        output[1][bin_id] += (LinearAlgebra.dot(vx[i] - vy[j], s_vector) / norm)
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



function pairwise_vel_cellist(sample_1::Vector{SVector{3,Float32}},
                                    vel_1::Vector{SVector{3,Float32}}, 
                                    sample_2::Vector{SVector{3,Float32}},
                                    vel_2::Vector{SVector{3,Float32}},
                                    bin_edges::Vector,
                                    box_size::SVector{3},
                                    distance_cutoff::AbstractFloat,
                                    distance_normalizer::Vector{<:AbstractFloat},
                                    )
    
    n_bins = size(bin_edges)
    box = Box(box_size, distance_cutoff)
    cl = CellList(sample_1, sample_2, box)
    output = (zeros(n_bins),
            zeros(Int32, n_bins))
    map_pairwise!(
        (x, y, i, j, d2, output) -> map_function(x, y, i, j, d2, output, vel_1, vel_2, bin_edges, box_size, distance_normalizer),
        output, box, cl,
        reduce = reduce_hist,
        show_progress = true
    )

    return output[2], output[1]
end


precompile(pairwise_vel_cellist, (Vector{SVector{3,Float32}},
                                    Vector{SVector{3,Float32}}, 
                                    Vector{SVector{3,Float32}},
                                    Vector{SVector{3,Float32}},
                                    Vector,
                                    SVector{3},
                                    AbstractFloat,
                                    Vector{<:AbstractFloat},
                                    ))
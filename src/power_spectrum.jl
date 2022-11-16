function setup_box(pos_x, pos_y, pos_z, box_pad)

    data = (pos_x, pos_y, pos_z)
    box_min = SVector([minimum(d) for d in data]...) .- box_pad / 2
    box_max = SVector([maximum(d) for d in data]...) .+ box_pad / 2
    box_size = @SVector [maximum(box_max .- box_min) for _ in 1:3]

    box_size, box_min

end #func

function k_vec(field, box_size) 
    dims = [size(field)...]
    sample_rate = map(eltype(field), 2π .* dims ./ box_size)
    kx = rfftfreq(dims[1], sample_rate[1])
    ky = fftfreq(dims[2], sample_rate[2])
    kz = fftfreq(dims[3], sample_rate[3])
    (kx, ky, kz)
end #func

function x_vec(field::AbstractArray, box_size, box_min)
    dims = [size(field)...]
    cell_size = box_size ./ dims
    Tuple(map(T, collect(box_min[i] + 0.5 * cell_size[i]:cell_size[i]:box_min[i] + box_size[i])) for i in 1:3)
end #func


function interlacing!(field_r::AbstractArray, field_k::AbstractArray, data_pos, data_wt, order, box_size, box_min, grid_size, fft_plan, k⃗)
    cell_size = box_size ./ grid_size
    shifts = [eltype(field_r)(i) / order for i = 1:(order - 1)]
    ρ = zero(field_r)
    ρ_k = zero(field_k)
    for s in shifts
        # This allocates new positions, can be improved changing cic
        fill!(ρ,0)
        CosmoCorr.cic!(ρ, data_pos..., data_wt, box_size, box_min .- s .* cell_size; wrap = true)
        mul!(ρ_k, fft_plan, ρ)
        Threads.@threads for I in CartesianIndices(ρ_k)
            kc = k⃗[1][I[1]] * cell_size[1] + k⃗[2][I[2]] * cell_size[2] + k⃗[3][I[3]] * cell_size[3]
            field_k[I] = (field_k[I] + ρ_k[I] * exp(im * s * kc)) / order
        end #for
    end #for
    ldiv!(field_r, fft_plan, field_k)
    
    
end #func

function compensate!(field_k::AbstractArray, k⃗, box_size, grid_size, shot)
    cell_size = box_size ./ grid_size
    if shot == 0
        Threads.@threads for I in CartesianIndices(field_k)
            correction = (1 - 2. / 3 * sin(0.5 * k⃗[1][I[1]] / cell_size[1])^2)^0.5 * 
                         (1 - 2. / 3 * sin(0.5 * k⃗[2][I[2]] / cell_size[2])^2)^0.5 * 
                         (1 - 2. / 3 * sin(0.5 * k⃗[3][I[3]] / cell_size[3])^2)^0.5
            field_k[I] /= correction
        end #for
    else 
        Threads.@threads for I in CartesianIndices(field_k)
            correction = sinc(0.5 * k⃗[1][I[1]] / cell_size[1] / π )^2 *
                         sinc(0.5 * k⃗[2][I[2]] / cell_size[2] / π )^2 * 
                         sinc(0.5 * k⃗[3][I[3]] / cell_size[3] / π )^2
            field_k[I] /= correction
        end #for
    end #if
    field_k
end #func

@with_kw mutable struct Mesh

    field_r
    field_k
    field_r_ell
    field_k_ell
    field_k_aux
    box_size
    box_min
    wdata
    wrand
    ndata
    nrand
    shot
    norm
    alpha
    box_pad
    interlacing::Int
    fft_plan

    # Simulation boxes w/o randoms
    function Mesh(data_pos, data_wt, grid_size, box_size, box_min, interlacing; fft_plan = nothing)
        field_r = zeros(eltype(data_pos[1]), grid_size...)
        CosmoCorr.cic!(field_r, data_pos..., data_wt, box_size, box_min; wrap = true)
        wdata = sum(data_wt)
        vol = box_size[1] * box_size[2] * box_size[3]
        shot = vol / wdata
        norm = wdata^2 / vol
        ndata = size(data_pos[1], 1)
        fft_plan = fft_plan == nothing ? plan_rfft(field_r) : fft_plan
        k⃗ = k_vec(field_r, box_size) 
        field_k = fft_plan * field_r
        if interlacing > 1
            interlacing!(field_r, field_k, data_pos, data_wt, interlacing, box_size, box_min, grid_size, fft_plan, k⃗)
        end #if

        
        new(field_r, field_k, nothing, nothing, nothing, box_size, box_min, wdata, zero(typeof(wdata)), ndata, zero(typeof(ndata)), shot, norm, zero(typeof(wdata)), zero(box_size), interlacing, fft_plan)
    end #func

    # Simulation boxes w/randoms
    function Mesh(data_pos, data_wt, rand_pos, rand_wt, grid_size, box_size, box_min, interlacing; fft_plan = nothing)
        data_field_r = zeros(eltype(data_pos[1]), grid_size...)
        rand_field_r = zero(data_field_r)
        CosmoCorr.cic!(data_field_r, data_pos..., data_wt, box_size, box_min; wrap = true)
        CosmoCorr.cic!(rand_field_r, rand_pos..., rand_wt, box_size, box_min; wrap = true)
        wdata = zero(eltype(data_wt))
        wrand = zero(eltype(data_wt))
        sumw2_0_dat = zero(eltype(data_wt))
        sumw2_0_ran = zero(eltype(rand_wt))
        sumw2_1_dat = zero(eltype(data_wt))
        sumw2_1_ran = zero(eltype(rand_wt))
        for i = axes(data_wt)
            wdata += data_wt
            new_data_wt[i] = data_wt[i] * data_fkp[i]
            sumw2_0_dat += new_data_wt[i]^2
            sumw2_1_dat += data_wt[i] * data_fkp[i]^2 * data_nz[i]
        end #for
        for i = axes(rand_wt)
            wrand += rand_wt
            new_rand_wt[i] = rand_wt[i] * rand_fkp[i]
            sumw2_0_ran += new_rand_wt[i]^2
            sumw2_1_ran += rand_wt[i] * rand_fkp[i]^2 * rand_nz[i]
        end #for
        alpha[k] = wdata / wrand
        shot = sumw2_0_dat + alpha * alpha * sumw2_0_ran
        if (sumw2_1_dat == 0)
            norm = alpha * sumw2_1_ran
        elseif (sumw2_1_ran == 0)
            norm = sumw2_1_dat
        else
            norm = alpha * sumw2_1_ran
        end #if
        ndata = size(data_pos[1], 1)
        nrand = size(rand_pos[1], 1)
        fft_plan = fft_plan == nothing ? plan_rfft(field_r) : fft_plan
        k⃗ = k_vec(field_r, box_size) 
        field_k = fft_plan * field_r
        if interlacing > 1
            interlacing!(field_r, field_k, data_pos, data_wt .* data_fkp, interlacing, box_size, box_min, grid_size, fft_plan, k⃗)
        end #if
        mul!(field_k, fft_plan, field_r_ell)
        if interlacing > 1
            interlacing!(field_r_ell, field_k, rand_pos, rand_wt .* rand_fkp, interlacing, box_size, box_min, grid_size, fft_plan, k⃗)
        end #if
        @. field_r .-= (alpha .* field_r_ell)
        @. field_r_ell .= field_r
        new(field_r, field_k, field_r_ell, nothing, nothing, box_size, box_min, wdata, wrand, ndata, nrand, shot, norm, alpha, zero(box_size), interlacing, fft_plan)
    end #func

    # Survey volumes
    function Mesh(data_pos, data_wt, data_fkp, data_nz, 
                    rand_pos, rand_wt, rand_fkp, rand_nz, 
                    grid_size, box_size, box_min, interlacing
                    ; fft_plan = nothing)
        box_size, box_min = setup_box(rand_pos..., box_pad)
        field_r = zeros(eltype(data_pos[1]), grid_size...)
        field_r_ell = zero(data_storage)
        CosmoCorr.cic!(field_r, data_pos..., data_wt .* data_fkp, box_size, box_min; wrap = false)
        CosmoCorr.cic!(field_r_ell, rand_pos..., rand_wt .* rand_fkp, box_size, box_min; wrap = false)

        wdata = zero(eltype(data_wt))
        wrand = zero(eltype(data_wt))
        sumw2_0_dat = zero(eltype(data_wt))
        sumw2_0_ran = zero(eltype(rand_wt))
        sumw2_1_dat = zero(eltype(data_wt))
        sumw2_1_ran = zero(eltype(rand_wt))
        for i = axes(data_wt)
            wdata += data_wt
            new_data_wt[i] = data_wt[i] * data_fkp[i]
            sumw2_0_dat += new_data_wt[i]^2
            sumw2_1_dat += data_wt[i] * data_fkp[i]^2 * data_nz[i]
        end #for
        for i = axes(rand_wt)
            wrand += rand_wt
            new_rand_wt[i] = rand_wt[i] * rand_fkp[i]
            sumw2_0_ran += new_rand_wt[i]^2
            sumw2_1_ran += rand_wt[i] * rand_fkp[i]^2 * rand_nz[i]
        end #for
        alpha[k] = wdata / wrand
        shot = sumw2_0_dat + alpha * alpha * sumw2_0_ran
        if (sumw2_1_dat == 0)
            norm = alpha * sumw2_1_ran
        elseif (sumw2_1_ran == 0)
            norm = sumw2_1_dat
        else
            norm = alpha * sumw2_1_ran
        end #if

        ndata = size(data_pos[1], 1)
        nrand = size(rand_pos[1], 1)
        fft_plan = fft_plan == nothing ? plan_rfft(field_r) : fft_plan
        k⃗ = k_vec(field_r, box_size) 
        field_k = fft_plan * field_r
        if interlacing > 1
            interlacing!(field_r, field_k, data_pos, data_wt .* data_fkp, interlacing, box_size, box_min, grid_size, fft_plan, k⃗)
        end #if
        mul!(field_k, fft_plan, field_r_ell)
        if interlacing > 1
            interlacing!(field_r_ell, field_k, rand_pos, rand_wt .* rand_fkp, interlacing, box_size, box_min, grid_size, fft_plan, k⃗)
        end #if
        @. field_r .-= (alpha .* field_r_ell)
        @. field_r_ell .= field_r
        new(field_r, field_k, field_r_ell, nothing, nothing, box_size, box_min, wdata, wrand, ndata, nrand, shot, norm, alpha, box_min, interlacing, fft_plan)
        
    end #func
end #struct

radius(x, y, z) = sqrt(x^2 + y^2 + z^2)
theta(x, y, z) = acos(z / radius(x, y, z))
phi(x, y, z) = atan(y, x)

sph_harm(m, l, ϕ, θ) = sf_legendre_sphPlm(l, m, cos(θ)*cis(m*ϕ))

for ell = 1:6
    name = Symbol("dens_k$(ell)")
    @eval begin
        function $(name)(mesh::Mesh, x⃗, k⃗)
            fill!(mesh.field_r_ell, 0)
            for m = -ell:1:ell
                @tturbo for i = axes(mesh.field_r_ell, 1), j = axes(mesh.field_r_ell, 2), k = axes(mesh.field_r_ell, 3)
                    mesh.field_r_ell[i,j,k] = mesh.field_r_ell[i,j,k] * sph_harm(m, ell, phi(x⃗[1][i], x⃗[2][j], x⃗[3][k]), theta(x⃗[1][i], x⃗[2][j], x⃗[3][k]))
                end #for
                mul!(mesh.field_k_aux, mesh.fft_plan, mesh.field_r_ell)
                Threads.@threads for I in CartesianIndices(mesh.field_k_ell)
                    if k⃗[2][I[2]] == 0 || k⃗[3][I[3]] == 0
                        mesh.field_k_ell[I] += mesh.field_k_aux[I]
                    else
                        mesh.field_k_ell[I] += mesh.field_k_aux[I] * sph_harm(m, ell, phi(k⃗[1][I[1]], k⃗[2][I[2]], k⃗[3][I[3]]), theta(k⃗[1][I[1]], k⃗[2][I[2]], k⃗[3][I[3]]))
                    end #if
                end #for
            end #for m
        end #func
    end #eval
end #for

function alias_corr(pik, interlaced)
    if interlaced
        fac = (1 / sinc(pik / π))^2
        return fac^2 #(for CIC only)
    else
        fac = sin(pik)^2
        return 1 / (1 - 0x1.5555555555555p-1 * fac)
    end #if
end #func

function count_modes_lin_sim(field_k_a::AbstractArray, field_k_b::AbstractArray, k_edges, k⃗, poles, los, interlaced, box_size, grid_size)
    dk = k_edges[2] - k_edges[1]
    power_poles = [zeros(real(eltype(field_k_a)), size(poles, 1), size(k_edges, 1)-1) for _ in 1:Threads.nthreads()]
    lcounts = [zeros(real(eltype(field_k_a)), size(poles, 1), size(k_edges, 1)-1) for _ in 1:Threads.nthreads()]
    counts = [zeros(Int, size(k_edges, 1)-1) for _ in 1:Threads.nthreads()]
    nbins = length(counts[1])
    cell_size = box_size ./ grid_size
    Threads.@threads for I in CartesianIndices(field_k_a)
        k² = k⃗[1][I[1]]^2 + k⃗[2][I[2]]^2 + k⃗[3][I[3]]^2
        kmod = sqrt(k²)
        (kmod > k_edges[end] || kmod < k_edges[1] || kmod == 0) && continue
        
        # k = 0                                     k = k_ny
        if ((kmod != 0) && (k⃗[1][I[1]] == 0)) || ((grid_size[3] & 1 == 0) && (I[1] == grid_size[3] / 2))
            num = 1
        else # 0 < k < k_ny
            num = 2
        end #if
            
        alias = alias_corr(0.5k⃗[1][I[1]] * cell_size[1], interlaced) * 
                alias_corr(0.5k⃗[2][I[2]] * cell_size[2], interlaced) * 
                alias_corr(0.5k⃗[3][I[3]] * cell_size[3], interlaced)
        kbin = (Int ∘ floor)((kmod - k_edges[1]) / dk) + 1
        (kbin < 1 || kbin > nbins) && continue
        mu = (k⃗[1][I[1]] * los[1] + k⃗[2][I[2]] * los[2] + k⃗[3][I[3]] * los[3]) / kmod
        p = real(field_k_a[I] * conj(field_k_b[I])) * num * alias
        println
        #if (I[1] == 1) && (I[2] == 1) && (I[3] == 1) 
        #    @show [0.5k⃗[n][I[n]] * cell_size[n] for n in 1:3]
        #    @show field_k_a[I] * conj(field_k_b[I]), field_k_a[I] 
        #    @show p, alias
        #    #(field_k_a[I] * conj(field_k_b[I]), field_k_a[I]) = (2.631019f6 + 0.0f0im, -769.7499f0 - 1427.762f0im)
        #    #p = 5.262038f6 + 0.0f0im
#
        #    #no compensate
        #    #(field_k_a[I] * conj(field_k_b[I]), field_k_a[I]) = (2.630801f6 + 0.0f0im, -769.71875f0 - 1427.7024f0im)
        #    #p = 5.261602f6 + 0.0f0im
        #end #if
        counts[Threads.threadid()][kbin] += num
        for (i, ell) in enumerate(poles)
            if ((ell & 1) == 1) && (num == 2)  # if ell odd
                continue
            end #if
            leg = Pl(mu, ell)
            power_poles[Threads.threadid()][i, kbin] += p * leg 
            lcounts[Threads.threadid()][i, kbin] += num * leg 
        end #for
    end #for

    if Threads.nthreads() > 1
        for i in 2:Threads.nthreads()
            power_poles[1] .+= power_poles[i]
            lcounts[1] .+= lcounts[i]
            counts[1] .+= counts[i]
        end #for
    end #if

    power_poles[1], lcounts[1], counts[1]
end #func


function count_modes(meshes::AbstractVector{Mesh}, poles, k_edges, is_sim, k⃗, los)
    if is_sim
        println("Counting modes...")
        p_counts, lcounts, counts = count_modes_lin_sim(meshes[1].field_k, meshes[2].field_k, k_edges, k⃗, poles, los, meshes[1].interlacing > 1, meshes[1].box_size, size(meshes[1].field_r))
        println("Done")
        auto = meshes[1] === meshes[2]
        for i = eachindex(poles), j = 1:length(k_edges)-1
            if counts[j] > 0
                p_counts[i,j] /= sqrt(meshes[1].norm * meshes[2].norm) * counts[j]
                p_counts[i,j] -= auto ? meshes[1].shot * lcounts[i,j] / counts[j] : 0
            end #if
            p_counts[i,j] *= (2 * poles[i] + 1)
        end #for
    end #if
    p_counts, counts
end #func
count_modes(mesh::Mesh, poles, k_edges, is_sim, k⃗, los) = count_modes([mesh, mesh], poles, k_edges, is_sim, k⃗, los)
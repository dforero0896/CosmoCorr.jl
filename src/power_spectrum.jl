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


function interlacing!(field_r::AbstractArray, field_k::AbstractArray, data_pos, data_wt, order, box_size, box_min, grid_size, fft_plan, k⃗, assign)
    cell_size = box_size ./ grid_size
    shifts = [eltype(field_r)(i) / order for i = 1:(order - 1)]
    ρ = zero(field_r)
    ρ_k = zero(field_k)
    for s in shifts
        # This allocates new positions, can be improved changing cic
        fill!(ρ,0)
        if assign == :cic
            CosmoCorr.cic!(ρ, data_pos..., data_wt, box_size, box_min .- s .* cell_size; wrap = true)
        elseif assign == :ngp
            CosmoCorr.ngp!(ρ, data_pos..., data_wt, box_size, box_min .- s .* cell_size; wrap = true)
        end #if
        mul!(ρ_k, fft_plan, ρ)
        Threads.@threads for I in CartesianIndices(ρ_k)
            kc = k⃗[1][I[1]] * cell_size[1] + k⃗[2][I[2]] * cell_size[2] + k⃗[3][I[3]] * cell_size[3]
            field_k[I] = (field_k[I] + ρ_k[I] * exp(im * s * kc)) / order
        end #for
    end #for
    ldiv!(field_r, fft_plan, field_k)
    
    
end #func

function compensate!(field_k::AbstractArray, k⃗, box_size, grid_size, shot)
    error("Function not corrected for different mass assignments")
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
    assign

    # Simulation boxes w/o randoms
    function Mesh(data_pos, data_wt, grid_size, box_size, box_min, interlacing, assign; fft_plan = nothing)
        data_pos = deepcopy(data_pos)
        field_r = zeros(eltype(data_pos[1]), grid_size...)
        if assign == :cic
            CosmoCorr.cic!(field_r, data_pos..., data_wt, box_size, box_min; wrap = true)
        elseif assign == :ngp
            CosmoCorr.ngp!(field_r, data_pos..., data_wt, box_size, box_min; wrap = true)
        else
            error("Mass assignment not recognized")
        end #if
        wdata = sum(data_wt)
        vol = box_size[1] * box_size[2] * box_size[3]
        shot = vol / wdata
        norm = wdata^2 / vol
        ndata = size(data_pos[1], 1)
        fft_plan = fft_plan == nothing ? plan_rfft(field_r) : fft_plan
        k⃗ = k_vec(field_r, box_size) 
        field_k = fft_plan * field_r
        if interlacing > 1
            interlacing!(field_r, field_k, data_pos, data_wt, interlacing, box_size, box_min, grid_size, fft_plan, k⃗, assign)
        end #if
        new(field_r, field_k, nothing, nothing, nothing, box_size, box_min, wdata, zero(typeof(wdata)), ndata, zero(typeof(ndata)), shot, norm, zero(typeof(wdata)), zero(box_size), interlacing, fft_plan, assign)
    end #func

    # Simulation boxes w/o randoms from array
    function Mesh(field_r::AbstractArray{T,3}, box_size, box_min, assign; fft_plan = nothing) where T <: Real
        wdata = sum(field_r)
        vol = box_size[1] * box_size[2] * box_size[3]
        shot = vol / wdata
        norm = wdata^2 / vol
        ndata = wdata
        fft_plan = fft_plan == nothing ? plan_rfft(field_r) : fft_plan
        field_k = fft_plan * field_r
        new(field_r, field_k, nothing, nothing, nothing, box_size, box_min, wdata, zero(typeof(wdata)), ndata, zero(typeof(ndata)), shot, norm, zero(typeof(wdata)), zero(box_size), 0, fft_plan, assign)
    end #func

    # Simulation boxes w/randoms
    function Mesh(data_pos, data_wt, rand_pos, rand_wt, grid_size, box_size, box_min, interlacing, assign; fft_plan = nothing)
        data_field_r = zeros(eltype(data_pos[1]), grid_size...)
        rand_field_r = zero(data_field_r)
        if assign == :cic
            CosmoCorr.cic!(data_field_r, data_pos..., data_wt, box_size, box_min; wrap = true)
            CosmoCorr.cic!(rand_field_r, rand_pos..., rand_wt, box_size, box_min; wrap = true)
        elseif assign == :ngp
            CosmoCorr.ngp!(data_field_r, data_pos..., data_wt, box_size, box_min; wrap = true)
            CosmoCorr.ngp!(rand_field_r, rand_pos..., rand_wt, box_size, box_min; wrap = true)
        else
            error("Mass assignment not recognized")
        end #if
        
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
            interlacing!(field_r, field_k, data_pos, data_wt .* data_fkp, interlacing, box_size, box_min, grid_size, fft_plan, k⃗, assign)
        end #if
        mul!(field_k, fft_plan, field_r_ell)
        if interlacing > 1
            interlacing!(field_r_ell, field_k, rand_pos, rand_wt .* rand_fkp, interlacing, box_size, box_min, grid_size, fft_plan, k⃗, assign)
        end #if
        @. field_r .-= (alpha .* field_r_ell)
        @. field_r_ell .= field_r
        new(field_r, field_k, field_r_ell, nothing, nothing, box_size, box_min, wdata, wrand, ndata, nrand, shot, norm, alpha, zero(box_size), interlacing, fft_plan, assign)
    end #func

    # Survey volumes
    function Mesh(data_pos, data_wt, data_fkp, data_nz, 
                    rand_pos, rand_wt, rand_fkp, rand_nz, 
                    grid_size, box_size, box_min, interlacing,
                    assign
                    ; fft_plan = nothing)
        box_size, box_min = setup_box(rand_pos..., box_pad)
        field_r = zeros(eltype(data_pos[1]), grid_size...)
        field_r_ell = zero(data_storage)
        if assign == :cic
            CosmoCorr.cic!(field_r, data_pos..., data_wt .* data_fkp, box_size, box_min; wrap = false)
            CosmoCorr.cic!(field_r_ell, rand_pos..., rand_wt .* rand_fkp, box_size, box_min; wrap = false)
        elseif assign == :ngp
            CosmoCorr.ngp!(field_r, data_pos..., data_wt .* data_fkp, box_size, box_min; wrap = false)
            CosmoCorr.ngp!(field_r_ell, rand_pos..., rand_wt .* rand_fkp, box_size, box_min; wrap = false)
        else
            error("Mass assignment not recognized")
        end #if
        

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
            interlacing!(field_r, field_k, data_pos, data_wt .* data_fkp, interlacing, box_size, box_min, grid_size, fft_plan, k⃗, assign)
        end #if
        mul!(field_k, fft_plan, field_r_ell)
        if interlacing > 1
            interlacing!(field_r_ell, field_k, rand_pos, rand_wt .* rand_fkp, interlacing, box_size, box_min, grid_size, fft_plan, k⃗, assign)
        end #if
        @. field_r .-= (alpha .* field_r_ell)
        @. field_r_ell .= field_r
        new(field_r, field_k, field_r_ell, nothing, nothing, box_size, box_min, wdata, wrand, ndata, nrand, shot, norm, alpha, box_min, interlacing, fft_plan, assign)
        
    end #func
end #struct
function update_mesh_mesh!(mesh::Mesh, field_r::AbstractArray{T,3}) where T <: Real
    vol = mesh.shot * mesh.wdata
    mesh.wdata = sum(field_r)
    mesh.shot = vol / mesh.wdata
    mesh.norm = mesh.wdata^2 / vol
    mesh.ndata = mesh.wdata
    mesh.field_r .= field_r
    mul!(mesh.field_k, mesh.fft_plan, mesh.field_r)
    mesh
end #func
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

function alias_corr(pik, interlaced; assign = :cic)
    if interlaced
        fac = (1 / sinc(pik / π))^2
        if assign == :cic
            return fac^2
        elseif assign == :ngp
            return fac 
        elseif assign == :tsc
            return fac^3
        end #if
    else
        fac = sin(pik)^2
        if assign == :cic
            return 1 / (1 - 0x1.5555555555555p-1 * fac)
        elseif assign == :ngp
            return 1
        elseif assign ==:tsc
            return 1 / (1 - fac + 0x1.1111111111111p-3 * fac * fac)
        end #if
    end #if
end #func

function count_modes_lin_sim(field_k_a::AbstractArray, field_k_b::AbstractArray, k_edges, k⃗, poles, los, interlaced, box_size, grid_size, assign)
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
            
        alias = alias_corr(0.5k⃗[1][I[1]] * cell_size[1], interlaced; assign = assign) * 
                alias_corr(0.5k⃗[2][I[2]] * cell_size[2], interlaced; assign = assign) * 
                alias_corr(0.5k⃗[3][I[3]] * cell_size[3], interlaced; assign = assign)
        kbin = (Int ∘ floor)((kmod - k_edges[1]) / dk) + 1
        (kbin < 1 || kbin > nbins) && continue
        mu = (k⃗[1][I[1]] * los[1] + k⃗[2][I[2]] * los[2] + k⃗[3][I[3]] * los[3]) / kmod
        p = real(field_k_a[I] * conj(field_k_b[I])) * num * alias
   
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
        p_counts, lcounts, counts = count_modes_lin_sim(meshes[1].field_k, meshes[2].field_k, k_edges, k⃗, poles, los, meshes[1].interlacing > 1, meshes[1].box_size, size(meshes[1].field_r), meshes[1].assign)
        auto = meshes[1] === meshes[2]
        for i = eachindex(poles), j = 1:length(k_edges)-1
            if counts[j] > 0
                p_counts[i,j] /= sqrt(meshes[1].norm * meshes[2].norm) * counts[j]
                p_counts[i,j] -= auto ? meshes[1].shot * lcounts[i,j] / counts[j] : 0
            end #if
            p_counts[i,j] *= (2 * poles[i] + 1)
        end #for
    else
        error("No power spectrum mode counting for survey data has been implemented")
    end #if
    p_counts, counts
end #func
count_modes(mesh::Mesh, poles, k_edges, is_sim, k⃗, los) = count_modes([mesh, mesh], poles, k_edges, is_sim, k⃗, los)


function power_spectrum(meshes::AbstractVector{Mesh}, poles, k_edges, is_sim, los)
    mul!(meshes[1].field_k, meshes[1].fft_plan, meshes[1].field_r)
    mul!(meshes[2].field_k, meshes[2].fft_plan, meshes[2].field_r)
    k⃗ = CosmoCorr.k_vec(meshes[1].field_r, box_size) 
    power, modes = CosmoCorr.count_modes(meshes, poles, k_edges, is_sim, k⃗, los)
    k = 0.5 .* (k_edges[2:end] .+ k_edges[1:end-1])
    k, power
end #function
function power_spectrum(mesh::Mesh, poles, k_edges, is_sim, los)
    mul!(mesh.field_k, mesh.fft_plan, mesh.field_r)
    k⃗ = CosmoCorr.k_vec(mesh.field_r, box_size) 
    power, modes = CosmoCorr.count_modes(mesh, poles, k_edges, is_sim, k⃗, los)
    k = 0.5 .* (k_edges[2:end] .+ k_edges[1:end-1])
    k, power, modes
end #function
function power_spectrum(field_r::AbstractArray{T,3}, box_size, box_min, poles, k_edges, los; assign = :ngp) where T<:Real
    mesh = Mesh(field_r, box_size, box_min, assign; fft_plan = nothing)
    k⃗ = CosmoCorr.k_vec(mesh.field_r, box_size) 
    power, modes = CosmoCorr.count_modes(mesh, poles, k_edges, true, k⃗, los)
    k = 0.5 .* (k_edges[2:end] .+ k_edges[1:end-1])
    k, power, modes
end #func

function power_spectrum(field_r_a::AbstractArray{T,3}, field_r_b::AbstractArray{T,3}, box_size, box_min, poles, k_edges, los; assign = :ngp) where T<:Real
    mesh_a = Mesh(field_r_a, box_size, box_min, assign; fft_plan = nothing)
    mesh_b = Mesh(field_r_b, box_size, box_min, assign; fft_plan = mesh_a.fft_plan)
    k⃗ = CosmoCorr.k_vec(mesh_a.field_r, box_size) 
    power, modes = CosmoCorr.count_modes([mesh_a, mesh_b], poles, k_edges, true, k⃗, los)
    k = 0.5 .* (k_edges[2:end] .+ k_edges[1:end-1])
    k, power, modes
end #func


function power_spectrum(pos::AbstractVector{<:AbstractVector{T}}, w::AbstractVector{T}, poles, k_edges, los, grid_size, box_size, box_min, interlacing; assign = :cic) where T <: Real
    mesh = CosmoCorr.Mesh(pos, w, grid_size, box_size, box_min, interlacing, assign)
    mul!(mesh.field_k, mesh.fft_plan, mesh.field_r)
    k⃗ = CosmoCorr.k_vec(mesh.field_r, box_size) 
    power, modes = CosmoCorr.count_modes(mesh, poles, k_edges, true, k⃗, los)
    k = 0.5 .* (k_edges[2:end] .+ k_edges[1:end-1])
    k, power, modes
end #func
power_spectrum(catalog::AbstractArray{T,2}, w::AbstractVector{T}, poles, k_edges, los, grid_size, box_size, box_min, interlacing; assign = :cic) where T <: Real = power_spectrum([@view(catalog[i,:]) for i=1:3], w::AbstractVector{T}, poles, k_edges, los, grid_size, box_size, box_min, interlacing; assign = assign)

function power_spectrum(pos_a::AbstractVector{<:AbstractVector{T}}, w_a::AbstractVector{T}, pos_b::AbstractVector{<:AbstractVector{T}}, w_b::AbstractVector{T}, poles, k_edges, los, grid_size, box_size, box_min, interlacing; assign = :cic) where T <: Real
    mesha = CosmoCorr.Mesh(pos_a, w_a, grid_size, box_size, box_min, interlacing, assign)
    mul!(mesha.field_k, mesha.fft_plan, mesha.field_r)
    meshb = CosmoCorr.Mesh(pos_b, w_b, grid_size, box_size, box_min, interlacing, assign; fft_plan = mesha.fft_plan)
    mul!(meshb.field_k, meshb.fft_plan, meshb.field_r)
    k⃗ = CosmoCorr.k_vec(mesha.field_r, box_size) 
    power, modes = CosmoCorr.count_modes([mesha, meshb], poles, k_edges, true, k⃗, los)
    k = 0.5 .* (k_edges[2:end] .+ k_edges[1:end-1])
    k, power, modes
end #func

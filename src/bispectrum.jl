function bispectrum_sim(field_k_a::AbstractArray, field_a::AbstractArray, k1, k2, theta, k⃗, interlaced, box_size, grid_size, fft_plan)
    @assert (grid_size[1] == grid_size[2]) && (grid_size[2] == grid_size[3])
    
    k_fund = 2 * pi ./ box_size
    field_k_1 = zero(field_k_a)
    field_k_2 = zero(field_k_a)
    field_k_3 = zero(field_k_a)
    I_k_1 = zero(field_k_a)
    I_k_2 = zero(field_k_a)
    I_k_3 = zero(field_k_a)

    field_1 = zero(field_a)
    field_2 = zero(field_a)
    field_3 = zero(field_a)
    I_1 = zero(field_a)
    I_2 = zero(field_a)
    I_3 = zero(field_a)

    bins = size(theta, 1)
    B = zeros(eltype(field_a), bins)
    Q = zeros(eltype(field_a), bins)
    triangles = zeros(eltype(field_a), bins)
    k_min = zeros(eltype(field_a), bins + 2)
    k_max = zeros(eltype(field_a), bins + 2)
    k_all = [k1, k2, sqrt.((k2 .* sin.(theta)).^2 + (k2 .* cos.(theta) .+ k1).^2)...]
    Pk = zeros(eltype(field_a), bins + 2)
    numbers = Vector{Vector{Int}}()

    for i = 1:bins+2
        push!(numbers, Vector{Int}())
        k_min[i] = (k_all[i] - k_fund[1]) / k_fund[1]
        k_max[i] = (k_all[i] + k_fund[1]) / k_fund[1]
    end #for
    
    cell_size = box_size ./ grid_size
    LI = LinearIndices(field_k_a)
    CI = CartesianIndices(field_k_a)
    for I in CI
        TI = Tuple(I)
                
        k² = k⃗[1][I[1]]^2 + k⃗[2][I[2]]^2 + k⃗[3][I[3]]^2
        kmod = sqrt(k²) / k_fund[1]
        
      
        ## k = 0                                     k = k_ny
        #if ((kmod != 0) && (k⃗[1][I[1]] == 0)) || ((grid_size[3] & 1 == 0) && (I[1] == grid_size[3] / 2))
        #    num = 1
        #else # 0 < k < k_ny
        #    num = 2
        #end #if
            
        alias = alias_corr(0.5k⃗[1][I[1]] * cell_size[1], interlaced) * 
                alias_corr(0.5k⃗[2][I[2]] * cell_size[2], interlaced) * 
                alias_corr(0.5k⃗[3][I[3]] * cell_size[3], interlaced)
        TI = Tuple(I)
        field_k_a[I] *= (alias)
        for i = 1:bins+2
            if (kmod > k_min[i]) && (kmod < k_max[i])
                push!(numbers[i], LI[I])
            end #if
        end #for
    end #for
    
    for i =1:length(numbers[1])
        I = CI[numbers[1][i]]
        field_k_1[I] = field_k_a[I]
        I_k_1[I] = 1.
    end #for
    ldiv!(field_1, fft_plan, field_k_1)
    ldiv!(I_1, fft_plan, I_k_1)
    Pk[1] = 0
    pairs = 0
    for I in CartesianIndices(field_1)
        Pk[1] += field_1[I]^2
        pairs += I_1[I]^2
    end #for
    Pk[1] = (Pk[1] / pairs) * (box_size[1] / grid_size[1]^2)^3

    for i =1:length(numbers[2])
        I = CI[numbers[2][i]]
        field_k_2[I] = field_k_a[I]
        I_k_2[I] = 1.
    end #for
    ldiv!(field_2, fft_plan, field_k_2)
    ldiv!(I_2, fft_plan, I_k_2)
    Pk[2] = 0
    pairs = 0
    for I in CartesianIndices(field_2)
        Pk[2] += field_2[I]^2
        pairs += I_2[I]^2
    end #for
    Pk[2] = (Pk[2] / pairs) * (box_size[1] / grid_size[1]^2)^3

    for j = 1:bins
        fill!(field_k_3, 0.)
        fill!(I_k_3, 0.)
        for i =1:length(numbers[j+2])
            I = CI[numbers[j+2][i]]
            field_k_3[I] = field_k_a[I]
            I_k_3[I] = 1.
        end #for
        ldiv!(field_3, fft_plan, field_k_3)
        ldiv!(I_3, fft_plan, I_k_3)
        Pk[j+2] = 0
        pairs = 0
        for I in CartesianIndices(field_3)
            Pk[j+2] += field_3[I]^2
            pairs += I_3[I]^2
        end #for
        Pk[j+2] = (Pk[j+2] / pairs) * (box_size[1] / grid_size[1]^2)^3

        B[j] = 0
        for I in CartesianIndices(field_3)
            B[j] += (field_1[I] * field_2[I] * field_3[I])
            triangles[j] += (I_1[I] * I_2[I] * I_3[I])
        end #for
        B[j] = (B[j] / triangles[j]) * (box_size[1]^2 / grid_size[1]^3)^3
        Q[j] = B[j] / (Pk[1] * Pk[2] + Pk[1] * Pk[j+2] + Pk[2] * Pk[j+2])
    end #for

    theta, B, Q, Pk
end #func


function bispectrum(mesh::Mesh, k1, k2, theta, is_sim)
    k⃗ = CosmoCorr.k_vec(mesh.field_r, mesh.box_size) 
    if is_sim
        return bispectrum_sim(mesh.field_k, mesh.field_r, k1, k2, theta, k⃗, mesh.interlacing > 1, mesh.box_size, size(mesh.field_r), mesh.fft_plan)
    else
        error("No implementation for data bispectrum")
    end #if
end #func

function bispectrum(pos::AbstractArray{<:AbstractArray{T}}, w::AbstractVector{T}, k1, k2, theta, is_sim, grid_size, box_size, box_min, interlacing) where T <:Real
    mesh = CosmoCorr.Mesh(pos, w, grid_size, box_size, box_min, interlacing)
    mul!(mesh.field_k, mesh.fft_plan, mesh.field_r)
    bispectrum(mesh, k1, k2, theta, is_sim)
end #func

bispectrum(catalog::AbstractArray{T,2}, w::AbstractVector{T}, k1, k2, theta, grid_size, box_size, box_min, is_sim, interlacing) where T <:Real = bispectrum([@view(catalog[i,:]) for i=1:3], w, k1, k2, theta, grid_size, box_size, box_min, is_sim, interlacing)
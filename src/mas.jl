

function cic(data_x::Array, data_y::Array, data_z::Array, data_w::Array, box_size::Array, box_min::Array, rho::Array)

    data_size = size(data_x)[1]
    n_bins = size(rho)
    @Threads.threads for i in range(1,data_size)
        x::Float32 = (data_x[i] - box_min[1]) * n_bins[1] / box_size[1] 
        y::Float32 = (data_y[i] - box_min[2]) * n_bins[2] / box_size[2] 
        z::Float32 = (data_z[i] - box_min[3]) * n_bins[3] / box_size[3] 

        x0::Int = Int(floor(x)) + 1
        y0::Int = Int(floor(y)) + 1
        z0::Int = Int(floor(z)) + 1

        wx1::Float32 = x - x0
        wx0::Float32 = 1 - wx1
        wy1::Float32 = y - y0
        wy0::Float32 = 1 - wy1
        wz1::Float32 = z - z0
        wz0::Float32 = 1 - wz1

        x0 = (x0 == n_bins[1]+1) ? 1 : x0
        y0 = (y0 == n_bins[2]+1) ? 1 : y0
        z0 = (z0 == n_bins[3]+1) ? 1 : z0

        x1 = (x0 == n_bins[1]) ? 1 : x0 + 1
        y1 = (y0 == n_bins[2]) ? 1 : y0 + 1
        z1 = (z0 == n_bins[3]) ? 1 : z0 + 1

        wx0 *= data_w[i]
        wx1 *= data_w[i]

        
        rho[x0,y0,z0] += wx0 * wy0 * wz0
        rho[x1,y0,z0] += wx1 * wy0 * wz0
        rho[x0,y1,z0] += wx0 * wy1 * wz0
        rho[x0,y0,z1] += wx0 * wy0 * wz1
        rho[x1,y1,z0] += wx1 * wy1 * wz0
        rho[x1,y0,z1] += wx1 * wy0 * wz1
        rho[x0,y1,z1] += wx0 * wy1 * wz1
        rho[x1,y1,z1] += wx1 * wy1 * wz1

        

    end
    return 0

end
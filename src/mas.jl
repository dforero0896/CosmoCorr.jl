function cic!(ρ, data_x, data_y, data_z, data_w, box_size, box_min; wrap::Bool = true) 
    n_bins = size(ρ)
    for i in eachindex(data_x)

        if wrap
            x = (data_x[i] - box_min[1]) > box_size[1] ?  data_x[i] - box_size[1] : data_x[i]
            y = (data_y[i] - box_min[1]) > box_size[1] ?  data_y[i] - box_size[1] : data_y[i]
            z = (data_z[i] - box_min[1]) > box_size[1] ?  data_z[i] - box_size[1] : data_z[i]
        end #if

        x = (x - box_min[1]) * n_bins[1] / box_size[1] + 1
        y = (y - box_min[2]) * n_bins[2] / box_size[2] + 1
        z = (z - box_min[3]) * n_bins[3] / box_size[3] + 1
        
        x0::Int = Int(floor(x))
        y0::Int = Int(floor(y))
        z0::Int = Int(floor(z))

        wx1 = x - x0
        wx0 = 1 - wx1
        wy1 = y - y0
        wy0 = 1 - wy1
        wz1 = z - z0
        wz0 = 1 - wz1

        x0 = (x0 == n_bins[1]+1) ? 1 : x0
        y0 = (y0 == n_bins[2]+1) ? 1 : y0
        z0 = (z0 == n_bins[3]+1) ? 1 : z0


        x1 = (x0 == n_bins[1]) & wrap ? 1 : x0 + 1
        y1 = (y0 == n_bins[2]) & wrap ? 1 : y0 + 1
        z1 = (z0 == n_bins[3]) & wrap ? 1 : z0 + 1

        wx0 *= data_w[i]
        wx1 *= data_w[i]
        #@show x0,y0,z0
        #@show x1,y1,z1
        
        ρ[x0,y0,z0] += wx0 * wy0 * wz0
        ρ[x1,y0,z0] += wx1 * wy0 * wz0
        ρ[x0,y1,z0] += wx0 * wy1 * wz0
        ρ[x0,y0,z1] += wx0 * wy0 * wz1
        ρ[x1,y1,z0] += wx1 * wy1 * wz0
        ρ[x1,y0,z1] += wx1 * wy0 * wz1
        ρ[x0,y1,z1] += wx0 * wy1 * wz1
        ρ[x1,y1,z1] += wx1 * wy1 * wz1
    end #for
    ρ
end #func

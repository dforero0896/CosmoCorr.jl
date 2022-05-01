using Threads

function cic(data_x::Array, data_y::Array, data_z::Array, box_size::Array, box_min::Array, rho::Array)

    data_size = size(data_x)[0]
    @threads for i in range(data_size)
        x = data_x[i]
        y = data_y[i]
        z = data_z[i]

        println(x, y, z)

    end

end
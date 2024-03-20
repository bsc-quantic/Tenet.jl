# Figure example for Tenet.HyperindConverter function

function smooth_annotation!(f; color=Makie.RGBAf(110 // 256, 170 // 256, 250 // 256, 60 // 256), xlims=[-2, 2], ylims=[-2, 2], offset_x=0.0, offset_y=0.0, radius_x=1.0, radius_y=1.0, num_waves=5, fluctuation_amplitude=0.1, phase_shift=0.0)
    ax = Axis(f)
    hidedecorations!(ax)
    hidespines!(ax)

    # Define limits of the plot
    xlims!(ax, xlims...)
    ylims!(ax, ylims...)

    # Create a perturbed filled shape
    theta = LinRange(0, 2π, 100)

    fluctuations = fluctuation_amplitude .* sin.(num_waves .* theta .+ phase_shift)

    # Apply the fluctuations and radius scaling
    perturbed_radius_x = radius_x .+ fluctuations
    perturbed_radius_y = radius_y .+ fluctuations

    circle_points = [Point2f((perturbed_radius_x[i]) * cos(theta[i]) + offset_x,
                              (perturbed_radius_y[i]) * sin(theta[i]) + offset_y) for i in 1:length(theta)]

    poly!(ax, circle_points, color=color, closed=true)
end

begin
    set_theme!(resolution=(800,200))

    fig = Figure()

    data = zeros(Float64, 2, 2, 2, 2)
    for i in 1:2
        for j in 1:2
            for k in 1:2
                # In data the 1st-2th are diagonal
                data[i, i, j, k] = k
            end
        end
    end

    A = Tensor(data, (:i, :j, :k, :l))
    B = Tensor(rand(2, 2), (:i, :m))
    C = Tensor(rand(2, 2), (:j, :n))

    bg_blue = Makie.RGBAf(110 // 256, 170 // 256, 250 // 256, 50 // 256)
    orange = Makie.RGBf(240 // 256, 180 // 256, 100 // 256)
    red = Makie.RGBf(240 // 256, 90 // 256, 70 // 256)

    tn = TensorNetwork([A, B, C])
    reduced = transform(tn, Tenet.DiagonalReduction)

    smooth_annotation!(
        fig[1, 1];
        color = bg_blue,
        xlims = [-2, 2],
        ylims = [-2, 2],
        offset_x = -0.21,
        offset_y = -0.42,
        radius_x = 0.38,
        radius_y = 0.8,
        num_waves = 6,
        fluctuation_amplitude = 0.02,
        phase_shift = 0.0)
    plot!(fig[1, 1], tn, layout=Spring(iterations=1000, C=0.5, seed=100); node_color=[red, orange, orange, :black, :black,:black, :black])

    smooth_annotation!(
        fig[1, 2];
        color = bg_blue,
        xlims = [-2, 2],
        ylims = [-2, 2],
        offset_x = 0.1,
        offset_y = -0.35,
        radius_x = 0.38,
        radius_y = 1.1,
        num_waves = 5,
        fluctuation_amplitude = 0.02,
        phase_shift = 1.9)
    plot!(fig[1, 2], reduced, layout=Spring(iterations=1000, C=0.5, seed=100),  node_color=[orange, orange, red, :black, :black, :black, :black, :black])# Figure example for Tenet.HyperindConverter function

    Label(fig[1, 1, Bottom()], "Original Tensor Network")
    Label(fig[1, 2, Bottom()], "Transformed Tensor Network")

    fig
end

begin
    set_theme!(resolution=(800,200))

    fig = Figure()

    # create a tensor network where tensors B and D can be absorbed
    A = Tensor(rand(2, 2, 2, 2), (:i, :j, :k, :l))
    B = Tensor(rand(2, 2), (:i, :m))
    C = Tensor(rand(2, 2, 2), (:m, :n, :o))
    E = Tensor(rand(2, 2, 2, 2), (:o, :p, :q, :j))

    bg_blue = Makie.RGBAf(110 // 256, 170 // 256, 250 // 256, 50 // 256)
    orange = Makie.RGBf(240 // 256, 180 // 256, 100 // 256)
    red = Makie.RGBf(240 // 256, 90 // 256, 70 // 256)

    tn = TensorNetwork([A, B, C, E])
    reduced = transform(tn, Tenet.RankSimplification)

    smooth_annotation!(
        fig[1, 1];
        color = bg_blue,
        xlims = [-2, 2],
        ylims = [-2, 2],
        offset_x = -0.32,
        offset_y = -0.5,
        radius_x = 0.25,
        radius_y = 0.94,
        num_waves = 6,
        fluctuation_amplitude = 0.01,
        phase_shift = 0.0)
    plot!(fig[1, 1], tn, layout=Spring(iterations=1000, C=0.5, seed=20); node_color=[orange, red, orange, orange, :black, :black, :black, :black, :black])

    smooth_annotation!(
        fig[1, 2];
        color = bg_blue,
        xlims = [-2, 2],
        ylims = [-2, 2],
        offset_x = 0.12,
        offset_y = -0.62,
        radius_x = 0.18,
        radius_y = 0.46,
        num_waves = 5,
        fluctuation_amplitude = 0.01,
        phase_shift = 0)
    plot!(fig[1, 2], reduced, layout=Spring(iterations=1000, C=0.5, seed=1); node_color=[red, orange, orange, :black, :black, :black, :black, :black])# Figure example for Tenet.HyperindConverter function

    Label(fig[1, 1, Bottom()], "Original Tensor Network")
    Label(fig[1, 2, Bottom()], "Transformed Tensor Network")

    fig
end

begin
    set_theme!(resolution=(800,200))

    fig = Figure()

    d = 2  # size of indices

    data = zeros(Float64, d, d, d, d, d)
    data2 = zeros(Float64, d, d, d)
    for i in 1:d
        for j in 1:d
            for k in 1:d
                # In data_anti the 1st-4th and 2nd-5th indices are antidiagonal
                data[i, j, k, d-i+1, d-j+1] = k
                data[j, i, k, d-j+1, d-i+1] = k + 2

                data2[i, d-i+1, k] = 1  # 1st-2nd indices are antidiagonal in data2_anti
            end
        end
    end

    A = Tensor(data, (:i, :j, :k, :l, :m))
    B = Tensor(data2, (:j, :n, :o))
    C = Tensor(rand(d, d, d), (:k, :p, :q))

    bg_blue = Makie.RGBAf(110 // 256, 170 // 256, 250 // 256, 50 // 256)
    orange = Makie.RGBf(240 // 256, 180 // 256, 100 // 256)
    red = Makie.RGBf(240 // 256, 90 // 256, 70 // 256)

    tn = TensorNetwork([A, B, C])
    gauged = transform(tn, Tenet.AntiDiagonalGauging)

    smooth_annotation!(
        fig[1, 1];
        color = bg_blue,
        xlims = [-2, 2],
        ylims = [-2, 2],
        offset_x = -0.42,
        offset_y = -0.5,
        radius_x = 0.25,
        radius_y = 0.84,
        num_waves = 6,
        fluctuation_amplitude = 0.01,
        phase_shift = 0.0)
    plot!(fig[1, 1], tn, layout=Spring(iterations=1000, C=0.5, seed=6); node_color=[orange, orange, red, :black, :black, :black, :black, :black, :black, :black])

    smooth_annotation!(
        fig[1, 2];
        color = bg_blue,
        xlims = [-2, 2],
        ylims = [-2, 2],
        offset_x = 0.12,
        offset_y = -0.62,
        radius_x = 0.18,
        radius_y = 0.46,
        num_waves = 5,
        fluctuation_amplitude = 0.01,
        phase_shift = 0)
    plot!(fig[1, 2], gauged, layout=Spring(iterations=1000, C=0.5, seed=6); node_color=[orange, orange, red, :black, :black, :black, :black, :black, :black, :black])

    Label(fig[1, 1, Bottom()], "Original Tensor Network")
    Label(fig[1, 2, Bottom()], "Transformed Tensor Network")

    fig
end


begin
    set_theme!(resolution=(800,200))

    fig = Figure()

    data = rand(3, 3, 3)
    data[:, 1:2, :] .= 0

    A = Tensor(data, (:i, :j, :k))
    B = Tensor(rand(3, 3), (:j, :l))
    C = Tensor(rand(3, 3), (:l, :m))

    bg_blue = Makie.RGBAf(110 // 256, 170 // 256, 250 // 256, 50 // 256)
    orange = Makie.RGBf(240 // 256, 180 // 256, 100 // 256)
    red = Makie.RGBf(240 // 256, 90 // 256, 70 // 256)

    tn = TensorNetwork([A, B, C])
    reduced = transform(tn, Tenet.ColumnReduction)

    smooth_annotation!(
        fig[1, 1];
        color = bg_blue,
        xlims = [-2, 2],
        ylims = [-2, 2],
        offset_x = -1.12,
        offset_y = -0.22,
        radius_x = 0.35,
        radius_y = 0.84,
        num_waves = 4,
        fluctuation_amplitude = 0.02,
        phase_shift = 0.0)
    plot!(fig[1, 1], tn, layout=Spring(iterations=1000, C=0.5, seed=6); node_color=[red, orange, orange, :black, :black, :black])

    smooth_annotation!(
        fig[1, 2];
        color = bg_blue,
        xlims = [-2, 2],
        ylims = [-2, 2],
        offset_x = -0.64,
        offset_y = 1.2,
        radius_x = 0.32,
        radius_y = 0.78,
        num_waves = 5,
        fluctuation_amplitude = 0.02,
        phase_shift = 0)

    Label(fig[1, 1, Bottom()], "Original Tensor Network")
    Label(fig[1, 2, Bottom()], "Transformed Tensor Network")
    plot!(fig[1, 2], reduced, layout=Spring(iterations=2000, C=40, seed=8); node_color=[red, orange, orange, :black, :black, :black])

    fig
end



begin
    set_theme!(resolution=(800,200))

    fig = Figure()

    v1 = Tensor([1, 2, 3], (:i,))
    v2 = Tensor([4, 5, 6], (:j,))
    m1 = Tensor(rand(3, 3), (:k, :l))

    t1 = contract(v1, v2)
    tensor = contract(t1, m1) # Define a tensor which can be splitted in three

    bg_blue = Makie.RGBAf(110 // 256, 170 // 256, 250 // 256, 50 // 256)
    orange = Makie.RGBf(240 // 256, 180 // 256, 100 // 256)
    red = Makie.RGBf(240 // 256, 90 // 256, 70 // 256)

    tn = TensorNetwork([tensor, Tensor(rand(3, 3, 3), (:k, :m, :n)), Tensor(rand(3, 3, 3), (:l, :n, :o))])
    reduced = transform(tn, Tenet.SplitSimplification)

    smooth_annotation!(
        fig[1, 1];
        color = bg_blue,
        xlims = [-2, 2],
        ylims = [-2, 2],
        offset_x = 0.24,
        offset_y = 0.6,
        radius_x = 0.32,
        radius_y = 0.78,
        num_waves = 5,
        fluctuation_amplitude = 0.015,
        phase_shift = 0.0)
    plot!(fig[1, 1], tn, layout=Spring(iterations=10000, C=0.5, seed=12); node_color=[red, orange, orange, :black, :black, :black, :black])

    smooth_annotation!(
        fig[1, 2];
        color = bg_blue,
        xlims = [-2, 2],
        ylims = [-2, 2],
        offset_x = -0.2,
        offset_y = -0.4,
        radius_x = 1.1,
        radius_y = 0.75,
        num_waves = 3,
        fluctuation_amplitude = 0.18,
        phase_shift = 0.8)

    Label(fig[1, 1, Bottom()], "Original Tensor Network")
    Label(fig[1, 2, Bottom()], "Transformed Tensor Network")
    plot!(fig[1, 2], reduced, layout=Spring(iterations=10000, C=13, seed=151); node_color=[orange, orange, red, red, red, :black, :black, :black, :black])

    fig
end

begin
    using QuacIO
    set_theme!(resolution=(800,400))

    sites_ = [5, 6, 14, 15, 16, 17, 24, 25, 26, 27, 28, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 61, 62, 63, 64, 65, 66, 67, 72, 73, 74, 75, 76, 83, 84, 85, 94]
    circuit = QuacIO.parse("./docs/src/examples/sycamore_53_10_0.qasm", format=QuacIO.Qflex(), sites=sites_)
    tn = TensorNetwork(circuit)
    transformed_tn = transform(tn, Tenet.RankSimplification)

    fig = Figure() # hide
    ax1 = Axis(fig[1, 1]) # hide
    p1 = plot!(ax1, tn; edge_width=0.75, node_size=8., node_attr=(strokecolor=:black, strokewidth=0.5)) # hide
    ax2 = Axis(fig[1, 2]) # hide
    p2 = plot!(ax2, transformed_tn; edge_width=0.75, node_size=8., node_attr=(strokecolor=:black, strokewidth=0.5))
    ax1.titlesize=20 # hide
    ax2.titlesize=20 # hide
    hidedecorations!(ax1)
    hidespines!(ax1)
    hidedecorations!(ax2)
    hidespines!(ax2)

    Label(fig[1, 1, Bottom()], "Original Tensor Network")
    Label(fig[1, 2, Bottom()], "Transformed Tensor Network")

    fig # hide
end
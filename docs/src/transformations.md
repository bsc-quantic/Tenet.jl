# Transformations

```@setup plot
using Makie
Makie.inline!(true)

using CairoMakie
using Tenet
using NetworkLayout

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

bg_blue = Makie.RGBAf(110 // 256, 170 // 256, 250 // 256, 50 // 256)
orange = Makie.RGBf(240 // 256, 180 // 256, 100 // 256)
red = Makie.RGBf(240 // 256, 90 // 256, 70 // 256)
```

In tensor network computations, it is good practice to apply various transformations to simplify the network structure, reduce computational cost, or prepare the network for further operations. These transformations modify the network's structure locally by permuting, contracting, factoring or truncating tensors.

A crucial reason why these methods are indispensable lies in their ability to drastically reduce the problem size of the contraction path search and also the contraction. This doesn't necessarily involve reducing the maximum rank of the Tensor Network itself, but more importantly, it reduces the size (or rank) of the involved tensors.

Our approach is based in [gray2021hyper](@cite), which can also be found in [quimb](https://quimb.readthedocs.io/).

In Tenet, we provide a set of predefined transformations which you can apply to your `TensorNetwork` using both the `transform`/`transform!` functions.

```@docs
transform
transform!
```

## Transformations

### Hyperindex converter

```@docs
Tenet.HyperindConverter
```

### Diagonal reduction

```@docs
Tenet.DiagonalReduction
```

```@example plot
set_theme!(resolution=(800,200)) # hide
fig = Figure() #hide

data = zeros(Float64, 2, 2, 2, 2) #hide
for i in 1:2 #hide
    for j in 1:2 #hide
        for k in 1:2 #hide
            data[i, i, j, k] = k #hide
        end #hide
    end #hide
end #hide

A = Tensor(data, (:i, :j, :k, :l)) #hide
B = Tensor(rand(2, 2), (:i, :m)) #hide
C = Tensor(rand(2, 2), (:j, :n)) #hide

tn = TensorNetwork([A, B, C]) #hide
reduced = transform(tn, Tenet.DiagonalReduction) #hide

smooth_annotation!( #hide
    fig[1, 1]; #hide
    color = bg_blue, #hide
    xlims = [-2, 2], #hide
    ylims = [-2, 2], #hide
    offset_x = -0.21, #hide
    offset_y = -0.42, #hide
    radius_x = 0.38, #hide
    radius_y = 0.8, #hide
    num_waves = 6, #hide
    fluctuation_amplitude = 0.02, #hide
    phase_shift = 0.0) #hide
plot!(fig[1, 1], tn, layout=Spring(iterations=1000, C=0.5, seed=100); node_color=[red, orange, orange, :black, :black,:black, :black]) #hide

smooth_annotation!( #hide
    fig[1, 2]; #hide
    color = bg_blue, #hide
    xlims = [-2, 2], #hide
    ylims = [-2, 2], #hide
    offset_x = 0.1, #hide
    offset_y = -0.35, #hide
    radius_x = 0.38, #hide
    radius_y = 1.1, #hide
    num_waves = 5, #hide
    fluctuation_amplitude = 0.02, #hide
    phase_shift = 1.9) #hide
plot!(fig[1, 2], reduced, layout=Spring(iterations=1000, C=0.5, seed=100),  node_color=[orange, orange, red, :black, :black, :black, :black, :black]) #hide

Label(fig[1, 1, Bottom()], "Original") #hide
Label(fig[1, 2, Bottom()], "Transformed") #hide

fig #hide
```

### Anti-diagonal reduction

```@docs
Tenet.AntiDiagonalGauging
```

### Rank simplification

```@docs
Tenet.RankSimplification
```

```@example plot
set_theme!(resolution=(800,200)) # hide
fig = Figure() #hide

A = Tensor(rand(2, 2, 2, 2), (:i, :j, :k, :l)) #hide
B = Tensor(rand(2, 2), (:i, :m)) #hide
C = Tensor(rand(2, 2, 2), (:m, :n, :o)) #hide
E = Tensor(rand(2, 2, 2, 2), (:o, :p, :q, :j)) #hide

tn = TensorNetwork([A, B, C, E]) #hide
reduced = transform(tn, Tenet.RankSimplification) #hide

smooth_annotation!( #hide
    fig[1, 1]; #hide
    color = bg_blue, #hide
    xlims = [-2, 2], #hide
    ylims = [-2, 2], #hide
    offset_x = -0.32, #hide
    offset_y = -0.5, #hide
    radius_x = 0.25, #hide
    radius_y = 0.94, #hide
    num_waves = 6, #hide
    fluctuation_amplitude = 0.01, #hide
    phase_shift = 0.0) #hide
plot!(fig[1, 1], tn, layout=Spring(iterations=1000, C=0.5, seed=20); node_color=[orange, red, orange, orange, :black, :black, :black, :black, :black]) #hide

smooth_annotation!( #hide
    fig[1, 2]; #hide
    color = bg_blue, #hide
    xlims = [-2, 2], #hide
    ylims = [-2, 2], #hide
    offset_x = 0.12, #hide
    offset_y = -0.62, #hide
    radius_x = 0.18, #hide
    radius_y = 0.46, #hide
    num_waves = 5, #hide
    fluctuation_amplitude = 0.01, #hide
    phase_shift = 0) #hide
plot!(fig[1, 2], reduced, layout=Spring(iterations=1000, C=0.5, seed=1); node_color=[red, orange, orange, :black, :black, :black, :black, :black]) #hide

Label(fig[1, 1, Bottom()], "Original") #hide
Label(fig[1, 2, Bottom()], "Transformed") #hide

fig #hide
```

### Column reduction

```@docs
Tenet.ColumnReduction
```

```@example plot
set_theme!(resolution=(800,200)) # hide
fig = Figure() #hide

data = rand(3, 3, 3) #hide
data[:, 1:2, :] .= 0 #hide

A = Tensor(data, (:i, :j, :k)) #hide
B = Tensor(rand(3, 3), (:j, :l)) #hide
C = Tensor(rand(3, 3), (:l, :m)) #hide

tn = TensorNetwork([A, B, C]) #hide
reduced = transform(tn, Tenet.ColumnReduction) #hide

smooth_annotation!( #hide
    fig[1, 1]; #hide
    color = bg_blue, #hide
    xlims = [-2, 2], #hide
    ylims = [-2, 2], #hide
    offset_x = -1.12, #hide
    offset_y = -0.22, #hide
    radius_x = 0.35, #hide
    radius_y = 0.84, #hide
    num_waves = 4, #hide
    fluctuation_amplitude = 0.02, #hide
    phase_shift = 0.0) #hide
plot!(fig[1, 1], tn, layout=Spring(iterations=1000, C=0.5, seed=6); node_color=[red, orange, orange, :black, :black, :black]) #hide

smooth_annotation!( #hide
    fig[1, 2]; #hide
    color = bg_blue, #hide
    xlims = [-2, 2], #hide
    ylims = [-2, 2], #hide
    offset_x = -0.64, #hide
    offset_y = 1.2, #hide
    radius_x = 0.32, #hide
    radius_y = 0.78, #hide
    num_waves = 5, #hide
    fluctuation_amplitude = 0.02, #hide
    phase_shift = 0) #hide

Label(fig[1, 1, Bottom()], "Original") #hide
Label(fig[1, 2, Bottom()], "Transformed") #hide
plot!(fig[1, 2], reduced, layout=Spring(iterations=2000, C=40, seed=8); node_color=[red, orange, orange, :black, :black, :black]) #hide

fig #hide
```

### Split simplification

```@docs
Tenet.SplitSimplification
```

```@example plot
set_theme!(resolution=(800,200)) # hide
fig = Figure() #hide

v1 = Tensor([1, 2, 3], (:i,)) #hide
v2 = Tensor([4, 5, 6], (:j,)) #hide
m1 = Tensor(rand(3, 3), (:k, :l)) #hide

t1 = contract(v1, v2) #hide
tensor = contract(t1, m1)  #hide

tn = TensorNetwork([tensor, Tensor(rand(3, 3, 3), (:k, :m, :n)), Tensor(rand(3, 3, 3), (:l, :n, :o))]) #hide
reduced = transform(tn, Tenet.SplitSimplification) #hide

smooth_annotation!( #hide
    fig[1, 1]; #hide
    color = bg_blue, #hide
    xlims = [-2, 2], #hide
    ylims = [-2, 2], #hide
    offset_x = 0.24, #hide
    offset_y = 0.6, #hide
    radius_x = 0.32, #hide
    radius_y = 0.78, #hide
    num_waves = 5, #hide
    fluctuation_amplitude = 0.015, #hide
    phase_shift = 0.0) #hide
plot!(fig[1, 1], tn, layout=Spring(iterations=10000, C=0.5, seed=12); node_color=[red, orange,  orange, :black, :black, :black, :black]) #hide

smooth_annotation!( #hide
    fig[1, 2]; #hide
    color = bg_blue, #hide
    xlims = [-2, 2], #hide
    ylims = [-2, 2], #hide
    offset_x = -0.2, #hide
    offset_y = -0.4, #hide
    radius_x = 1.1, #hide
    radius_y = 0.75, #hide
    num_waves = 3, #hide
    fluctuation_amplitude = 0.18, #hide
    phase_shift = 0.8) #hide

Label(fig[1, 1, Bottom()], "Original") #hide
Label(fig[1, 2, Bottom()], "Transformed") #hide
plot!(fig[1, 2], reduced, layout=Spring(iterations=10000, C=13, seed=151); node_color=[orange, orange, red, red, red, :black, :black, :black, :black]) #hide

fig #hide
```

## Example: RQC simplification

Local transformations can dramatically reduce the complexity of tensor networks. Take as an example the Random Quantum Circuit circuit on the Sycamore chip from Google's quantum advantage experiment [arute2019quantum](@cite).

```@example plot
using QuacIO
set_theme!(resolution=(800,400)) # hide

sites = [5, 6, 14, 15, 16, 17, 24, 25, 26, 27, 28, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 61, 62, 63, 64, 65, 66, 67, 72, 73, 74, 75, 76, 83, 84, 85, 94]
circuit = QuacIO.parse(joinpath(@__DIR__, "sycamore_53_10_0.qasm"), format=QuacIO.Qflex(), sites=sites)
tn = QuantumTensorNetwork(circuit)

# Apply transformations to the tensor network
transformed_tn = transform(tn, [Tenet.AntiDiagonalGauging, Tenet.DiagonalReduction, Tenet.ColumnReduction, Tenet.RankSimplification])

fig = Figure() # hide
ax1 = Axis(fig[1, 1]) # hide
p1 = plot!(ax1, tn; edge_width=0.75, node_size=8., node_attr=(strokecolor=:black, strokewidth=0.5)) # hide
ax2 = Axis(fig[1, 2]) # hide
p2 = plot!(ax2, transformed_tn; edge_width=0.75, node_size=8., node_attr=(strokecolor=:black, strokewidth=0.5)) # hide
ax1.titlesize, ax2.titlesize = 20, 20 # hide
hidedecorations!(ax1) # hide
hidespines!(ax1) # hide
hidedecorations!(ax2) # hide
hidespines!(ax2) # hide

Label(fig[1, 1, Bottom()], "Original") # hide
Label(fig[1, 2, Bottom()], "Transformed") # hide

fig # hide
```

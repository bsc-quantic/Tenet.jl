# `Product` ansatz

```@setup examples
using Tenet
using Makie
Makie.inline!(true)
set_theme!(resolution=(800,400))
using GraphMakie
using CairoMakie
CairoMakie.activate!(type = "svg")
using NetworkLayout
```

A [`Product`](@ref) is the simplest Quantum Tensor Network [`Ansatz`](@ref), which consists of a one [`Tensor](@ref) per [`Site`](@ref) without any bonds, so all the sites are unconnected.
The [`Socket`](@ref) type of a [`Product`](@ref) (whether it represents a [`State`](@ref) or an [`Operator`](@ref)) depends on the rank of the tensors provided in the constructor.

## `Product` State
Each tensor is one-dimensional, with the only index being the output physical index.

```@example examples
fig = Figure() # hide

qtn = Product([rand(2) for _ in 1:3])

graphplot!(fig[1,1], qtn, layout=Spring(iterations=1000, C=0.5, seed=100)) # hide
Label(fig[1,1, Bottom()], "Product State") # hide

fig # hide
```

## `Product` Operator
Each tensor is two-dimensional, with the indices being the input and output physical indices.

```@example examples
fig = Figure() # hide

qtn = Product([rand(2, 2) for _ in 1:3])

graphplot!(fig[1,1], qtn, layout=Spring(iterations=1000, C=0.5, seed=100)) # hide
Label(fig[1,1, Bottom()], "Product Operator") # hide

fig #hide
```
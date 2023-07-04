# Projected Entangled Pair States (PEPS)

Projected Entangled Pair States (PEPS) are a Quantum Tensor Network ansatz whose tensors are laid out in a 2D lattice. Depending on the boundary conditions, the chains can be open or closed (i.e. periodic boundary conditions).

```@setup viz
using Makie
Makie.inline!(true)
set_theme!(resolution=(800,400))

using CairoMakie

using Tenet
using NetworkLayout
```

```@example viz
fig = Figure() # hide

tn_open = rand(PEPS{Open}, rows=10, cols=10, χ=4) # hide
tn_periodic = rand(PEPS{Periodic}, rows=10, cols=10, χ=4) # hide

plot!(fig[1,1], tn_open, layout=Stress(seed=1)) # hide
plot!(fig[1,2], tn_periodic, layout=Stress(seed=10,dim=2,iterations=100000)) # hide

Label(fig[1,1, Bottom()], "Open") # hide
Label(fig[1,2, Bottom()], "Periodic") # hide

fig # hide
```

```@example viz
fig = Figure() # hide

tn_open = rand(PEPO{Open}, rows=10, cols=10, χ=4) # hide
tn_periodic = rand(PEPO{Periodic}, rows=10, cols=10, χ=4) # hide

plot!(fig[1,1], tn_open, layout=Stress(seed=1)) # hide
plot!(fig[1,2], tn_periodic, layout=Stress(seed=10,dim=2,iterations=100000)) # hide

Label(fig[1,1, Bottom()], "Open") # hide
Label(fig[1,2, Bottom()], "Periodic") # hide

fig # hide
```

## Projected Entangled Pair Operators (PEPO)

```@docs
ProjectedEntangledPair
ProjectedEntangledPair(::Any)
```

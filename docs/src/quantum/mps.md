# Matrix Product States (MPS)

```@setup viz
using Makie
Makie.inline!(true)
set_theme!(resolution=(800,200))

using CairoMakie

using Tenet
using NetworkLayout
```

```@docs
MatrixProduct
MatrixProduct(::Any)
```

```@example viz
fig = Figure() # hide

tn_open = rand(MatrixProduct{State,Open}, n=10, χ=4) # hide
tn_periodic = rand(MatrixProduct{State,Periodic}, n=10, χ=4) # hide

plot!(fig[1,1], tn_open, layout=Spring(iterations=1000, C=0.5, seed=100)) # hide
plot!(fig[1,2], tn_periodic, layout=Spring(iterations=1000, C=0.5, seed=100)) # hide

Label(fig[1,1, Bottom()], "Open") # hide
Label(fig[1,2, Bottom()], "Periodic") # hide

fig # hide
```

## Matrix Product Operators (MPO)

```@example viz
fig = Figure() # hide

tn_open = rand(MatrixProduct{Operator,Open}, n=10, χ=4) # hide
tn_periodic = rand(MatrixProduct{Operator,Periodic}, n=10, χ=4) # hide

plot!(fig[1,1], tn_open, layout=Spring(iterations=1000, C=0.5, seed=100)) # hide
plot!(fig[1,2], tn_periodic, layout=Spring(iterations=1000, C=0.5, seed=100)) # hide

Label(fig[1,1, Bottom()], "Open") # hide
Label(fig[1,2, Bottom()], "Periodic") # hide

fig # hide
```

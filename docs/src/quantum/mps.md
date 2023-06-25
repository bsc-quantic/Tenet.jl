# Matrix Product States (MPS)

Matrix Product States (MPS) are a Quantum Tensor Network ansatz whose tensors are laid out in a 1D chain.
Due to this, these networks are also known as _Tensor Trains_ in other mathematical fields.
Depending on the boundary conditions, the chains can be open or closed (i.e. periodic boundary conditions).

```@setup viz
using Makie
Makie.inline!(true)
set_theme!(resolution=(800,200))

using CairoMakie

using Tenet
using NetworkLayout
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

Matrix Product Operators (MPO) are the operator version of [Matrix Product State (MPS)](#matrix-product-states-mps).
The major difference between them is that MPOs have 2 indices per site (1 input and 1 output) while MPSs only have 1 index per site (i.e. an output).

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

In `Tenet`, the generic `MatrixProduct` ansatz implements this topology. Type variables are used to address their functionality (`State` or `Operator`) and their boundary conditions (`Open` or `Periodic`).

```@docs
MatrixProduct
MatrixProduct(::Any)
```

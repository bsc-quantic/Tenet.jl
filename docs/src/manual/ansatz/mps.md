# Matrix Product States (MPS)

Matrix Product States (MPS) are a Quantum Tensor Network ansatz whose tensors are laid out in a 1D chain.
Due to this, these networks are also known as _Tensor Trains_ in other mathematical fields.
Depending on the boundary conditions, the chains can be open or closed (i.e. periodic boundary conditions).

```@setup viz
using Makie
Makie.inline!(true)
set_theme!(resolution=(800,200))

using CairoMakie
using GraphMakie

using Tenet
using NetworkLayout
```

```@example viz
tn = rand(MPS; n=10, maxdim=4) # hide
graphplot(tn, layout=Spring(iterations=1000, C=0.5, seed=100)) # hide
```

## Matrix Product Operators (MPO)

Matrix Product Operators (MPO) are the operator version of [Matrix Product State (MPS)](#matrix-product-states-mps).
The major difference between them is that MPOs have 2 indices per site (1 input and 1 output) while MPSs only have 1 index per site (i.e. an output).

```@example viz
tn = rand(MPO; n=10, maxdim=4) # hide
graphplot(tn, layout=Spring(iterations=1000, C=0.5, seed=100)) # hide
```

In `Tenet`, the generic `MatrixProduct` ansatz implements this topology. Type variables are used to address their functionality (`State` or `Operator`) and their boundary conditions (`Open` or `Periodic`).

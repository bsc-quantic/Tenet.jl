# Matrix Product States (MPS)

Matrix Product States ([`MPS`](@ref)) are a Quantum Tensor Network ansatz whose tensors are laid out in a 1D chain.
Due to this, these networks are also known as _Tensor Trains_ in other scientific fields.
Depending on the boundary conditions, the chains can be open or closed (i.e. periodic boundary conditions), currently
only `Open` boundary conditions are supported in `Tenet`.

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

open_mps = rand(MPS, n=10, χ=4) # hide

plot!(fig[1,1], open_mps, layout=Spring(iterations=1000, C=0.5, seed=100)) # hide

Label(fig[1,1, Bottom()], "Open") # hide

fig # hide
```

### Canonical Forms

A Matrix Product State ([`MPS`](@ref)) representation is not unique. Instead, a single `MPS` can be represented in different canonical forms. We can check the canonical form of an `MPS` by calling the [`form`](@ref) function.

```@example
mps = MPS([rand(2, 2), rand(2, 2, 2), rand(2, 2)])

form(mps)
```

Each canonical form can be useful in different situations, and the choice of the canonical form can affect the efficiency of the algorithms used to manipulate the `MPS`. Currently, `Tenet` supports the [`NonCanonical`](@ref), [`CanonicalForm`](@ref) and [`MixedCanonical`](@ref) forms.

#### `NonCanonical` Form
The default form of an `MPS` when we do not specify a canonical form.

#### `CanonicalForm`
Also known as Vidal's form. This form stores each `Tensor` of the `MPS` as a sequence of $\Gamma$ unitary tensors and $\lambda$ vectors:

```math
| \psi \rangle = \sum_{i_1, \dots, i_N} \Gamma_1^{i_1} \lambda_2^{i_2} \Gamma_2^{i_2} \dots \lambda_{N-1}^{i_{N-1}} \Gamma_{N-1}^{i_{N-1}} \lambda_N^{i_N} \Gamma_N^{i_N} | i_1, \dots, i_N \rangle \, .
```
This form can be obtained by calling [`canonize!`](@ref) on an `MPS`:

```@example
mps = MPS([rand(2, 2), rand(2, 2, 2), rand(2, 2)])
canonize!(mps)

form(mps)
```

#### `MixedCanonical` Form
This form stores the `Tensor`s in an `MPS` as left or right canonical wether the `Tensor` is on the left or right of the ortogonality center, which is stored in the field `orthog_center` of the `MixedCanonical` form.

```@example
mps = MPS([rand(2, 2), rand(2, 2, 2), rand(2, 2)])
mixed_canonize!(mps, Site(2))

form(mps)
```

## Matrix Product Operators (MPO)

Matrix Product Operators ([`MPO`](@ref)) are the operator version of [Matrix Product State (MPS)](#matrix-product-states-mps).
The major difference between them is that MPOs have 2 indices per site (1 input and 1 output) while MPSs only have 1 index per site (i.e. an output). Currently, only `Open` boundary conditions are supported in `Tenet`.

```@example viz
fig = Figure() # hide

open_mpo = rand(MatrixProduct{Operator,Open}, n=10, χ=4) # hide

plot!(fig[1,1], open_mpo, layout=Spring(iterations=1000, C=0.5, seed=100)) # hide

Label(fig[1,1, Bottom()], "Open") # hide

fig # hide
```

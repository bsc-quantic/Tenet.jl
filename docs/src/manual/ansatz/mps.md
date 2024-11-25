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

open_mps = rand(MPS; n=10, maxdim=4) # hide

plot!(fig[1,1], open_mps, layout=Spring(iterations=1000, C=0.5, seed=100)) # hide

Label(fig[1,1, Bottom()], "Open") # hide

fig # hide
```

### Canonical Forms

An `MPS` representation is not unique: a single `MPS` can be represented in different canonical [`Form`](@ref). The choice of canonical form can affect the efficiency and stability of algorithms used to manipulate the `MPS`. You can check the canonical form of an `MPS` by calling the `form` function:

```@example
mps = MPS([rand(2, 2), rand(2, 2, 2), rand(2, 2)])

form(mps)
```

Currently, `Tenet` supports the [`NonCanonical`](@ref), [`CanonicalForm`](@ref) and [`MixedCanonical`](@ref) forms.

#### `NonCanonical` Form
In the `NonCanonical` form, the tensors in the `MPS` do not satisfy any particular orthogonality conditions. This is the default `form` when an `MPS` is initialized without specifying a canonical form. It is useful for general purposes but may not be optimal for certain computations that benefit from orthogonality.

#### `Canonical` Form
Also known as Vidal's form, the `Canonical` form represents the `MPS` using a sequence of isometric tensors (`Γ`) and diagonal vectors (`λ`) containing the Schmidt coefficients. The `MPS` is expressed as:

```math
| \psi \rangle = \sum_{i_1, \dots, i_N} \Gamma_1^{i_1} \lambda_2 \Gamma_2^{i_2} \dots \lambda_{N-1} \Gamma_{N-1}^{i_{N-1}} \lambda_N \Gamma_N^{i_N} | i_1, \dots, i_N \rangle \, .
```

You can convert an `MPS` to the `Canonical` form by calling `canonize!`:

```@example
mps = MPS([rand(2, 2), rand(2, 2, 2), rand(2, 2)])
canonize!(mps)

form(mps)
```

#### `MixedCanonical` Form
In the `MixedCanonical` form, tensors to the left of the orthogonality center are left-canonical, tensors to the right are right-canonical, and the tensors at the orthogonality center (which can be `Site` or `Vector{<:Site}`) contains the entanglement information between the left and right parts of the chain. The position of the orthogonality center is stored in the `orthog_center` field.

You can convert an `MPS` to the `MixedCanonical` form and specify the orthogonality center using `mixed_canonize!`:

```@example
mps = MPS([rand(2, 2), rand(2, 2, 2), rand(2, 2)])
mixed_canonize!(mps, Site(2))

form(mps)
```

##### Additional Resources
For more in-depth information on Matrix Product States and their canonical forms, you may refer to:
- Schollwöck, U. (2011). The density-matrix renormalization group in the age of matrix product states. Annals of physics, 326(1), 96-192.


## Matrix Product Operators (MPO)

Matrix Product Operators ([`MPO`](@ref)) are the operator version of [Matrix Product State (MPS)](#matrix-product-states-mps).
The major difference between them is that MPOs have 2 indices per site (1 input and 1 output) while MPSs only have 1 index per site (i.e. an output). Currently, only `Open` boundary conditions are supported in `Tenet`.

```@example viz
fig = Figure() # hide

open_mpo = rand(MPO, n=10, maxdim=4) # hide

plot!(fig[1,1], open_mpo, layout=Spring(iterations=1000, C=0.5, seed=100)) # hide

Label(fig[1,1, Bottom()], "Open") # hide

fig # hide
```

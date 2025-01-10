# Matrix Product States (MPS)

```@setup plot
using Tenet
using Makie
Makie.inline!(true)
set_theme!(resolution=(800,200))

using CairoMakie
using GraphMakie
CairoMakie.activate!(type = "svg")

using NetworkLayout
```

Matrix Product States ([`MPS`](@ref)) are a Quantum Tensor Network [`Ansatz`](@ref) whose tensors are laid out in a 1D chain.
Due to this, these networks are also known as _Tensor Trains_ in other scientific fields.
Depending on the boundary conditions, these chains can be open or closed (i.e. periodic boundary conditions). Currently
only `Open` boundary conditions are supported in `Tenet`.

```@example plot
fig = Figure() # hide
open_mps = rand(MPS; n=10, maxdim=4)

plot!(fig[1,1], open_mps, layout=Spring(iterations=1000, C=0.5, seed=100)) # hide
Label(fig[1,1, Bottom()], "Open") # hide

fig # hide
```

The default ordering of the indices on the `MPS` constructor is (physical, left, right), but you can specify the ordering by passing the `order` keyword argument:

```@repl plot
mps = MPS([rand(4, 2), rand(4, 8, 2), rand(8, 2)]; order=[:l, :r, :o])
```
where `:l`, `:r`, and `:o` represent the left, right, and outer physical indices, respectively.


### Canonical Forms

An `MPS` representation is not unique: a single `MPS` can be represented in different canonical forms. The choice of canonical form can affect the efficiency and stability of algorithms used to manipulate the `MPS`.
The current form of the `MPS` is stored as the trait [`Form`](@ref) and can be accessed via the `form` function:

```@repl plot
mps = MPS([rand(2, 2), rand(2, 2, 2), rand(2, 2)])

form(mps)
```
> :warning: Depending on the form, `Tenet` will dispatch under the hood the appropriate algorithm which assumes full use of the canonical form, so be careful when making modifications that might alter the canonical form without changing the trait.

`Tenet` has the internal function [`Tenet.check_form`](@ref) to check if the `MPS` is in the correct canonical form. This function can be used to ensure that the `MPS` is in the correct form before performing any operation that requires it.
Currently, `Tenet` supports the [`NonCanonical`](@ref), [`CanonicalForm`](@ref) and [`MixedCanonical`](@ref) forms.

#### `NonCanonical` Form
In the `NonCanonical` form, the tensors in the `MPS` do not satisfy any particular orthogonality conditions. This is the default `form` when an `MPS` is initialized without specifying a canonical form. It is useful for general purposes but may not be optimal for certain computations that benefit from orthogonality.

#### `Canonical` Form
Also known as Vidal's form, the `Canonical` form represents the `MPS` using a sequence of isometric tensors (`Γ`) and diagonal vectors (`λ`) containing the Schmidt coefficients. The `MPS` is expressed as:

```math
| \psi \rangle = \sum_{i_1, \dots, i_N} \Gamma_1^{i_1} \lambda_2 \Gamma_2^{i_2} \dots \lambda_{N-1} \Gamma_{N-1}^{i_{N-1}} \lambda_N \Gamma_N^{i_N} | i_1, \dots, i_N \rangle \, .
```

You can convert an `MPS` to the `Canonical` form by calling `canonize!`:

```@repl plot
mps = MPS([rand(2, 2), rand(2, 2, 2), rand(2, 2)])
canonize!(mps)

form(mps)
```

#### `MixedCanonical` Form
In the `MixedCanonical` form, tensors to the left of the orthogonality center are left-canonical, tensors to the right are right-canonical, and the tensors at the orthogonality center (which can be `Site` or `Vector{<:Site}`) contains the entanglement information between the left and right parts of the chain. The position of the orthogonality center is stored in the `orthog_center` field.

You can convert an `MPS` to the `MixedCanonical` form and specify the orthogonality center using `mixed_canonize!`. Additionally, one can check that the `MPS` is effectively in mixed canonical form using the functions `isleftcanonical` and `isrightcanonical`, which return `true` if the `Tensor` at that particular site is left or right canonical, respectively.

```@repl plot
mps = MPS([rand(2, 2), rand(2, 2, 2), rand(2, 2)])
mixed_canonize!(mps, Site(2))

isisometry(mps, 1; dir=:right) # Check if the first tensor is left canonical
isisometry(mps, 3; dir=:left) # Check if the third tensor is right canonical

form(mps)
```

##### Additional Resources
For more in-depth information on Matrix Product States and their canonical forms, you may refer to:
- Schollwöck, U. (2011). The density-matrix renormalization group in the age of matrix product states. Annals of physics, 326(1), 96-192.


## Matrix Product Operators (MPO)

Matrix Product Operators ([`MPO`](@ref)) are the operator version of [Matrix Product State (MPS)](#matrix-product-states-mps).
The major difference between them is that MPOs have 2 indices per site (1 input and 1 output) while MPSs only have 1 index per site (i.e. an output). Currently, only `Open` boundary conditions are supported in `Tenet`.

```@example plot
fig = Figure() # hide
open_mpo = rand(MPO, n=10, maxdim=4)

plot!(fig[1,1], open_mpo, layout=Spring(iterations=1000, C=0.5, seed=100)) # hide
Label(fig[1,1, Bottom()], "Open") # hide

fig # hide
```

To apply an `MPO` to an `MPS`, you can use the `evolve!` function:

```@repl plot
mps = rand(MPS; n=6)
mpo = rand(MPO; n=6)

size.(tensors(mps))

evolve!(mps, mpo; normalize=false)

size.(tensors(mps))
```

As we can see, the bond dimension of the `MPS` has increased after applying the `MPO` to it.
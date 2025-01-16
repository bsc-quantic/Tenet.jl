# Matrix Product States (MPS)

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

Matrix Product States ([`MPS`](@ref)) (also known as _Tensor Trains_) are a Quantum Tensor Network [`Ansatz`](@ref) whose tensors are laid out in a 1D chain. Depending on the boundary conditions, these chains can be open or closed (i.e. periodic boundary conditions).

!!! warning
    Currently only [`Open`](@ref) boundary conditions are supported in Tenet.

```@example examples
fig = Figure() # hide
open_mps = rand(MPS; n=10, maxdim=4) # hide

graphplot!(fig[1,1], open_mps, layout=Spring(iterations=1000, C=0.5, seed=100)) # hide
Label(fig[1,1, Bottom()], "Open MPS") # hide

fig # hide
```

In Tenet, a Matrix Product State can be easily created by passing a list of arrays to the [`MPS`](@ref) constructor:
```@repl examples
ψ = MPS([rand(2, 2), rand(2, 2, 4), rand(2, 4, 2), rand(2, 2)])
```

The default ordering of the indices on the [`MPS`](@ref) constructor is (physical, left, right), but you can specify the ordering by passing the `order` keyword argument:

```@repl examples
ϕ = MPS([rand(2, 2), rand(2, 2, 4), rand(4, 2, 2), rand(2, 2)]; order=[:l, :o, :r])
```
where `:l`, `:r`, and `:o` represent the left, right, and outer physical indices, respectively.

Additionally, Tenet has the [`rand`](@ref) function to generate random [`MPS`](@ref) with a given number of sites and maximum bond dimension:

```@repl examples
Φ = rand(MPS, n=8, maxdim=10)
size.(tensors(Φ))
```

### Canonical Forms

An [`MPS`](@ref) representation is not unique: a single [`MPS`](@ref) can be represented in different canonical forms. The choice of canonical form can affect the efficiency and stability of algorithms used to manipulate the [`MPS`](@ref).
The current form of the [`MPS`](@ref) is stored as the trait [`Form`](@ref) and can be accessed via the [`form`](@ref) function:

```@repl examples
form(ψ)
```
!!! warning
    Depending on the form, Tenet will dispatch under the hood the appropriate algorithm which assumes full use of the canonical form, so be careful when making modifications that might alter the canonical form without changing the trait.

Tenet has the internal function [`Tenet.check_form`](@ref) to check if the `MPS` is in the correct canonical form. This function can be used to ensure that the `MPS` is in the correct form before performing any operation that requires it.
Currently, Tenet supports the [`NonCanonical`](@ref), [`CanonicalForm`](@ref) and [`MixedCanonical`](@ref) forms.

#### `NonCanonical` Form
In the [`NonCanonical`](@ref) form, the tensors in the [`MPS`](@ref) do not satisfy any particular orthogonality conditions. This is the default [`form`](@ref) when an [`MPS`](@ref) is initialized without specifying a canonical form. It is useful for general purposes but may not be optimal for certain computations that benefit from orthogonality.

#### `Canonical` Form
Also known as Vidal's form, the [`Canonical`](@ref) form represents the [`MPS`](@ref) using a sequence of tensors (`Γ`) and diagonal vectors (`λ`) containing the Schmidt coefficients. The [`MPS`](@ref) is expressed as:

```math
| \psi \rangle = \sum_{i_1, \dots, i_N} \Gamma_1^{i_1} \lambda_2 \Gamma_2^{i_2} \dots \lambda_{N-1} \Gamma_{N-1}^{i_{N-1}} \lambda_N \Gamma_N^{i_N} | i_1, \dots, i_N \rangle \, .
```

You can convert an `MPS` to the `Canonical` form by calling `canonize!`:

```@repl examples
canonize!(ψ)

form(ψ)
```

#### `MixedCanonical` Form
In the [`MixedCanonical`](@ref) form, tensors to the left of the orthogonality center are left-canonical, tensors to the right are right-canonical, and the tensors at the orthogonality center (which can be [`Site`](@ref) or `Vector{<:Site}`) contains the entanglement information between the left and right parts of the chain. In Tenet, a left (right) canonical `Tensor` is an isometry pointing to the direction right (left). The position of the orthogonality center is stored in the `orthog_center` field.

You can convert an [`MPS`](@ref) to the [`MixedCanonical`](@ref) form and specify the orthogonality center using [`mixed_canonize!`](@ref). Additionally, one can check that the `MPS` is effectively in mixed canonical form using the function [`isisometry`](@ref), which returns `True` if the [`Tensor`](@ref) at that particular site is an isometry pointing at direction `dir`:

```@repl examples
mixed_canonize!(ψ, Site(2))

isisometry(ψ, 1; dir=:right) # Check if the first tensor is left canonical
isisometry(ψ, 3; dir=:left) # Check if the third tensor is right canonical

form(ψ)
```

## Matrix Product Operators (MPO)

Matrix Product Operators ([`MPO`](@ref)) are the operator version of [Matrix Product State (MPS)](#matrix-product-states-mps).
The major difference between them is that MPOs have 2 indices per site (1 input and 1 output) while MPSs only have 1 index per site (i.e. an output). Currently, only [`Open`](@ref) boundary conditions are supported in Tenet.

```@example examples
fig = Figure() # hide
open_mpo = rand(MPO, n=10, maxdim=4)

graphplot!(fig[1,1], open_mpo, layout=Spring(iterations=1000, C=0.5, seed=100)) # hide
Label(fig[1,1, Bottom()], "Open") # hide

fig # hide
```

To apply an [`MPO`](@ref) to an [`MPS`](@ref), you can use the [`evolve!`](@ref) function:

```@repl examples
mps = rand(MPS; n=6)

evolve!(mps, open_mpo; normalize=false)
```

## Additional Resources
For more in-depth information on Matrix Product States and their canonical forms, you may refer to:
- Schollwöck, U. (2011). The density-matrix renormalization group in the age of matrix product states. Annals of physics, 326(1), 96-192.
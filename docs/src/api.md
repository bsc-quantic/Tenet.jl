# API

## `Tensor`

```@docs
Base.size(::Tensor)
Tenet.contract(::Tensor, ::Tensor)
LinearAlgebra.svd(::Tensor)
LinearAlgebra.qr(::Tensor)
LinearAlgebra.lu(::Tensor)
```

## `TensorNetwork`

```@docs
TensorNetwork
inds(::Tenet.TensorNetwork)
size(::Tenet.TensorNetwork)
tensors(::Tenet.TensorNetwork)
push!(::Tenet.TensorNetwork, ::Tensor)
append!(::Tenet.TensorNetwork, ::Base.AbstractVecOrTuple{<:Tensor})
merge!(::Tenet.TensorNetwork, ::Tenet.TensorNetwork)
pop!(::Tenet.TensorNetwork, ::Tensor)
delete!(::Tenet.TensorNetwork, ::Any)
replace!
selectdim
slice!
view(::Tenet.TensorNetwork)
Base.copy(::Tenet.TensorNetwork)
Base.rand(::Type{TensorNetwork}, n::Integer, regularity::Integer)
```

### Transformations

```@docs
transform
transform!
Tenet.HyperFlatten
Tenet.HyperGroup
Tenet.ContractSimplification
Tenet.DiagonalReduction
Tenet.AntiDiagonalGauging
Tenet.Truncate
Tenet.SplitSimplificationd
```

## `Quantum`

```@docs
Quantum
Tenet.TensorNetwork(::Quantum)
Base.adjoint(::Quantum)
sites
nsites
Tenet.inds(::Quantum; kwargs...)
Tenet.tensors(::Quantum; kwargs...)
inputs
outputs
lanes
ninputs
noutputs
nlanes
Socket
socket(::Quantum)
Scalar
State
Operator
Base.merge(::Quantum, ::Quantum...)
```

## `Ansatz`

## `Product`

## `MPS`

```@docs
MPS
```

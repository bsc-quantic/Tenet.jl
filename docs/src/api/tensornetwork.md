# TensorNetwork

```@docs
TensorNetwork
inds(::Tenet.TensorNetwork)
size(::Tenet.TensorNetwork)
tensors(::Tenet.TensorNetwork)
push!(::Tenet.TensorNetwork, ::Tensor)
pop!(::Tenet.TensorNetwork, ::Tensor)
append!(::Tenet.TensorNetwork, ::Base.AbstractVecOrTuple{<:Tensor})
merge!(::Tenet.TensorNetwork, ::Tenet.TensorNetwork)
delete!(::Tenet.TensorNetwork, ::Any)
replace!
selectdim
slice!
view(::Tenet.TensorNetwork)
Base.copy(::Tenet.TensorNetwork)
Base.rand(::Type{TensorNetwork}, n::Integer, regularity::Integer)
```

## Transformations

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

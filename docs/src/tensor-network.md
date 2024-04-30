# Tensor Networks

Tensor Networks (TN) are a graphical notation for representing complex multi-linear functions. For example, the following equation

```math
\sum_{ijklmnop} A_{im} B_{ijp} C_{njk} D_{pkl} E_{mno} F_{ol}
```

can be represented visually as

```@raw html
<figure>
<img width=500 src="assets/tn-sketch.svg" alt="Sketch of a Tensor Network">
<figcaption>Sketch of a Tensor Network</figcaption>
</figure>
```

The graph's nodes represent tensors and edges represent tensor indices.

In `Tenet`, these objects are represented by the [`TensorNetwork`](@ref) type.

```@docs
TensorNetwork
```

Information about a `TensorNetwork` can be queried with the following functions.

## Query information

```@docs
inds(::Tenet.TensorNetwork)
size(::Tenet.TensorNetwork)
tensors(::Tenet.TensorNetwork)
```

## Modification

### Add/Remove tensors

```@docs
push!(::Tenet.TensorNetwork, ::Tensor)
append!(::Tenet.TensorNetwork, ::Base.AbstractVecOrTuple{<:Tensor})
merge!(::Tenet.TensorNetwork, ::Tenet.TensorNetwork)
pop!(::Tenet.TensorNetwork, ::Tensor)
delete!(::Tenet.TensorNetwork, ::Any)
```

### Replace existing elements

```@docs
replace!
```

## Slicing

```@docs
selectdim
slice!
view(::Tenet.TensorNetwork)
```

## Miscelaneous

```@docs
Base.copy(::Tenet.TensorNetwork)
Base.rand(::Type{TensorNetwork}, n::Integer, regularity::Integer)
```

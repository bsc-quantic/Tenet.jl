# Tensor Networks

Tensor Networks (TN) are a graphical notation for representing complex multi-linear functions. For example, the following equation

```math
\sum_{ijklmnop} A_{im} B_{ijp} C_{njk} D_{pkl} E_{mno} F_{ol}
```

can be represented visually as

```@raw html
<figure>
<img width=500 src="../assets/tn-sketch.svg" alt="Sketch of a Tensor Network">
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
inds(::TensorNetwork)
size(::TensorNetwork)
tensors(::TensorNetwork)
length(::TensorNetwork)
```

## Modification

### Add/Remove tensors

```@docs
push!(::TensorNetwork, ::Tensor)
append!(::TensorNetwork, ::Base.AbstractVecOrTuple{<:Tensor})
pop!(::TensorNetwork, ::Tensor)
delete!(::TensorNetwork, ::Any)
```

### Replace existing elements

```@docs
replace
replace!
```

## Selection

```@docs
select
selectdim
slice!
view(::TensorNetwork)
```

## Miscelaneous

```@docs
Base.copy(::TensorNetwork)
```

```@docs
Base.rand(::Type{TensorNetwork}, n::Integer, regularity::Integer)
```

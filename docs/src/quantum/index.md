# Introduction

In `Tenet`, we define a [`Quantum`](@ref) Tensor Network as a [`TensorNetwork`](@ref) with a notion of sites and directionality.

```@docs
Quantum
```

## Plugs

```@docs
plug
```

## Sites

```@docs
sites
```

```@docs
tensors(::TensorNetwork{<:Quantum}, ::Integer)
```

## Adjoint

```@docs
adjoint
```

## Concatenation

```@docs
hcat(::TensorNetwork{<:Quantum}, ::TensorNetwork{<:Quantum})
```

## Bounds

```@docs
boundary
```

## Norm

```@docs
LinearAlgebra.norm(::TensorNetwork{<:Quantum}, p::Real)
LinearAlgebra.normalize!(::TensorNetwork{<:Quantum}, ::Real)
```

## Fidelity

```@docs
fidelity
```

# Introduction

In `Tenet`, we define a [`Quantum`](@ref) Tensor Network as a [`TensorNetwork`](@ref) with a notion of sites and directionality.

```@docs
Quantum
```

```@docs
plug
```

```@docs
sites
```

```@docs
tensors(::TensorNetwork{Quantum}, ::Integer)
```

```@docs
boundary
```

## Adjoint

```@docs
adjoint
```

## Concatenation

```@docs
hcat(::TensorNetwork{Quantum}, ::TensorNetwork{Quantum})
```

## Norm

```@docs
LinearAlgebra.norm(::TensorNetwork{Quantum}, p::Real)
LinearAlgebra.normalize!(::TensorNetwork{Quantum}, ::Real)
```

## Fidelity

```@docs
fidelity
```

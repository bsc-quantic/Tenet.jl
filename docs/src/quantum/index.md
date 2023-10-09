# Introduction

In `Tenet`, we define a [`QuantumTensorNetwork`](@ref) as a [`TensorNetwork`](@ref) with a notion of sites and directionality.

```@docs
QuantumTensorNetwork
plug
sites
```

## Adjoint

```@docs
adjoint
```

## Norm

```@docs
LinearAlgebra.norm(::Tenet.AbstractQuantumTensorNetwork, ::Real)
LinearAlgebra.normalize!(::Tenet.AbstractQuantumTensorNetwork, ::Real)
```

## Trace

```@docs
LinearAlgebra.tr(::Tenet.AbstractQuantumTensorNetwork)
```

## Fidelity

```@docs
fidelity
```

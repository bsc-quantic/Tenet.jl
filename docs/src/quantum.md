# `Quantum` Tensor Networks

```@docs
Quantum
Tenet.TensorNetwork(::Quantum)
Base.adjoint(::Quantum)
sites
nsites
```

## Queries

```@docs
Tenet.inds(::Quantum; kwargs...)
Tenet.tensors(::Quantum; kwargs...)
```

## Connecting `Quantum` Tensor Networks

```@docs
inputs
outputs
lanes
ninputs
noutputs
nlanes
```

```@docs
Socket
socket(::Quantum)
Scalar
State
Operator
```

```@docs
Base.merge(::Quantum, ::Quantum...)
```

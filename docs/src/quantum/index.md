# Introduction

!!! danger "🚧 Broken code 🚧"
    There is a lot of work in progress, and this code may not work yet.
    Take this code as an example of what we want to achieve.

```@docs
Quantum
```

```@docs
State
Operator
```

## Bounds

```@docs
bounds
```

## Sites

```@docs
sites
insites
outsites
```

```@docs
tensors(::TensorNetwork{<:Quantum}, ::Integer)
```

```@docs
siteinds
insiteinds
outsiteinds
physicalinds
virtualinds
```

## Adjoint

```@docs
adjoint
```

## Concatenation

```@docs
hcat(::TensorNetwork{<:Quantum}, ::TensorNetwork{<:Quantum})
```

## Norm

```@docs
LinearAlgebra.norm(::TensorNetwork{<:Quantum}, p::Real)
LinearAlgebra.normalize!(::TensorNetwork{<:Quantum})
```

## Fidelity

```@docs
fidelity
```

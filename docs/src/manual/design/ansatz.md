# Ansatz

```@setup examples
using Tenet
```

## The `Lattice` type

A [`Lattice`](@ref) is a graph whose vertices represent [`Site`](@ref)s and the edges represent the neighboring connectivity between them.

## The `Ansatz` type

A [`Ansatz`](@ref) is a [`Quantum`](@ref) Tensor Network that stores information about [`Site`](@ref) connectivity in a [`Lattice`](@ref).

## Canonization

⚠️ WIP

### The `Form` trait

[`Form`](@ref) dynamic trait represents the canonical form in which the [`Ansatz`](@ref) is right now. You can use the [`form`](@ref) function to consult it:

## Time Evolution

In some sense, it's like the state has evolved through the operator.

[`evolve!`] is a high-level wrapper for different methods used for time-evolution, but currently only the "Simple Update" algorithm is implemented in [`simple_update!`](@ref).

```@repl examples
evolve!
```

⚠️ WIP

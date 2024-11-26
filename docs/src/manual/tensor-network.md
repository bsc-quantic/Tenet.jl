# Tensor Networks

Tensor Networks (TN) are a graphical notation for representing complex multi-linear functions. For example, the following equation

```math
\sum_{ijklmnop} A_{im} B_{ijp} C_{njk} D_{pkl} E_{mno} F_{ol}
```

can be represented visually as

The graph's nodes represent tensors and edges represent tensor indices.

In `Tenet`, these objects are represented by the [`TensorNetwork`](@ref) type.

Information about a `TensorNetwork` can be queried with the following functions.

## Query information

## Modification

### Add/Remove tensors

### Replace existing elements

## Slicing

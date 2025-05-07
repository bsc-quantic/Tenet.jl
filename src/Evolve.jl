"""
    evolve!(tn, operator; threshold = nothing, maxdim = nothing, normalize = false, kwargs...)

Evolve (through time) a Tensor Network with a operator.

!!! note

    Currently only the "Simple Update" algorithm is implemented.

# Keyword Arguments

  - `threshold`: The threshold to truncate the bond dimension.
  - `maxdim`: The maximum bond dimension to keep.
  - `normalize`: Whether to normalize the state after truncation.

# Notes

  - The gate must act on neighboring sites according to the [`Lattice`](@ref) of the Tensor Network.
  - The gate must have the same number of inputs and outputs.
  - Currently only the "Simple Update" algorithm is used and the gate must be a 1-site or 2-site operator.
"""
function evolve! end

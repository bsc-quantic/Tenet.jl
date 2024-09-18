module Tenet

import EinExprs: inds

include("Helpers.jl")

include("Tensor.jl")
export Tensor, contract, dim, expand

include("Numerics.jl")

include("TensorNetwork.jl")
export TensorNetwork, tensors, arrays, neighbors, slice!
export contract, contract!, groupinds!

include("Transformations.jl")
export transform, transform!

include("Site.jl")
export Site, @site_str, isdual

include("Quantum.jl")
export Quantum, ninputs, noutputs, inputs, outputs, sites, nsites

include("Ansatz/Ansatz.jl")
export Ansatz
export socket, Scalar, State, Operator
export boundary, Open, Periodic
export form

export canonize_site, canonize_site!, canonize, canonize!, mixed_canonize, mixed_canonize!, truncate!
export evolve!, expect, overlap

# reexports from EinExprs
export einexpr, inds

end

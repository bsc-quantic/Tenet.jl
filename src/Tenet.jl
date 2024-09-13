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

include("Ansatz/Product.jl")
export Product

include("Ansatz/Dense.jl")
export Dense

# include("Ansatz/Chain.jl")
# export Chain

include("Ansatz/MPS.jl")
export MPS
export leftindex, rightindex, isleftcanonical, isrightcanonical
export canonize_site, canonize_site!, truncate!
export canonize, canonize!, mixed_canonize, mixed_canonize!

include("Ansatz/Grid.jl")
export Grid
export PEPS, pPEPS, PEPO, pPEPO

export evolve!, expect, overlap

# reexports from EinExprs
export einexpr, inds

end

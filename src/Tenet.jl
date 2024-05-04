module Tenet

import EinExprs: inds

include("Helpers.jl")

include("Tensor.jl")
export Tensor, contract, dim, expand

include("Numerics.jl")

include("TensorNetwork.jl")
export TensorNetwork, tensors, arrays, neighbors, slice!
export contract, contract!

include("Transformations.jl")
export transform, transform!

include("Compiler.jl")

# reexports from EinExprs
export einexpr, inds

end

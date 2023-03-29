module Tenet

using Requires: @require

include("Helpers.jl")
include("Numerics.jl")

import Tensors: Tensor, contract, labels, dim
export Tensor, contract, labels

include("Index.jl")
export Index, isphysical, isvirtual, site, ishyperind, links
export tags, tag!, untag!, hastag

include("TensorNetwork.jl")
export TensorNetwork, tensors, arrays, inds, openinds, hyperinds, select
export contract, contract!
export Ansatz, ansatz, Arbitrary

include("Transformations.jl")
export transform, transform!

include("Quantum/Quantum.jl")
export Quantum, bounds, Open, Closed, State, Operator
export physicalinds, virtualinds, sites, insites, insiteind, insiteinds, outsites, outsiteind, outsiteinds
export fidelity

export MatrixProductState
export MatrixProductOperator

include("Differentiation.jl")

function __init__()
    @require Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" include("Visualization.jl")
end

# reexports from OptimizedEinsum
export contractpath

# reexports from LinearAlgebra
export norm, normalize!

end
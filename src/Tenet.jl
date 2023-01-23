module Tenet

using Requires: @require

include("Helpers.jl")

include("Einsum.jl")
include("Tensor.jl")
export Tensor, labels, dim

include("Index.jl")
export Index, isphysical, isvirtual, site, ishyper, links
export tags, tag!, untag!, hastag

include("TensorNetwork.jl")
export TensorNetwork, tensors, arrays, inds, openinds, hyperinds, select, reindex!
export contract, contract!
export Ansatz, ansatz, Arbitrary

include("Transformations.jl")
export transform, transform!

include("Quantum.jl")
export Quantum
export physicalinds, virtualinds, sites, insites, insiteinds, outsites, outsiteinds

function __init__()
    @require Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" include("Visualization.jl")
end

# reexports from OptimizedEinsum
export contractpath

end
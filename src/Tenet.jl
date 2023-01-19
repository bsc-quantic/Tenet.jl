module Tenet

using Requires: @require

include("Index.jl")
export Index, isphysical, isvirtual, site, ishyper, links
export tags, tag!, untag!, hastag

include("Einsum.jl")
include("Tensor.jl")
export Tensor, labels, dim

include("TensorNetwork.jl")
export TensorNetwork, tensors, arrays, inds, openinds, hyperinds

include("GenericTensorNetwork.jl")
export GenericTensorNetwork

include("Quantum.jl")

function __init__()
    @require Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" include("Visualization.jl")
end

# reexports from OptimizedEinsum
export contractpath

end
module Tenet

using Requires: @require

include("Helpers.jl")
include("Numerics.jl")

import Tensors: Tensor, contract, labels, dim, tags, hastag, tag!, untag!

include("TensorNetwork.jl")
export TensorNetwork, tensors, arrays, select, selectdim!, splice!
export contract, contract!
export Ansatz, ansatz, Arbitrary

include("Transformations.jl")
export transform, transform!

include("Quantum/Quantum.jl")
export Quantum
export Boundary, boundary, Open, Periodic
export Plug, plug, Property, State, Operator
export sites, fidelity

export MatrixProduct

if !isdefined(Base, :get_extension)
    using Requires
end

@static if !isdefined(Base, :get_extension)
    function __init__()
        @require ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4" include("../ext/ChainRulesCoreExt.jl")
        @require Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" include("../ext/MakieExt.jl")
        @require Quac = "b9105292-1415-45cf-bff1-d6ccf71e6143" include("../ext/QuacExt.jl")
    end
end

# reexports from LinearAlgebra
export norm, normalize!

# reexports from Tensors
export Tensor, contract, labels, dim, tags, hastag, tag!, untag!

# TODO reexports from EinExprs

end

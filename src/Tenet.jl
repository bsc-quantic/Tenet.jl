module Tenet

include("Helpers.jl")

import Tensors: Tensor, contract, labels, dim, tags, hastag, tag!, untag!

include("TensorNetwork.jl")
export TensorNetwork, tensors, arrays, select, slice!
export contract, contract!
export Ansatz, ansatz, Arbitrary

include("Transformations.jl")
export transform, transform!

include("Quantum/Quantum.jl")
export Quantum
export Boundary, boundary, Open, Periodic, Infinite
export Plug, plug, Property, State, Operator
export sites, fidelity

export MatrixProduct
export ProjectedEntangledPair, PEPS, PEPO

if !isdefined(Base, :get_extension)
    using Requires
end

@static if !isdefined(Base, :get_extension)
    function __init__()
        @require Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" include("../ext/TenetMakieExt.jl")
        @require Quac = "b9105292-1415-45cf-bff1-d6ccf71e6143" include("../ext/TenetQuacExt.jl")
    end
end

# reexports from LinearAlgebra
export norm, normalize!

# reexports from Tensors
export Tensor, contract, labels, dim, tags, hastag, tag!, untag!

# reexports from EinExprs
export einexpr

end

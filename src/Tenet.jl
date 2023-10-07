module Tenet

import EinExprs: inds

include("Helpers.jl")

include("Tensor.jl")
export Tensor, contract, dim, expand

include("Numerics.jl")

include("TensorNetwork.jl")
export TensorNetwork, tensors, arrays, select, slice!
export contract, contract!

include("Transformations.jl")
export transform, transform!

include("Quantum/Quantum.jl")
export QuantumTensorNetwork, sites, fidelity
export Plug, plug, Property, State, Dual, Operator
export Boundary, boundary, Open, Periodic, Infinite

include("Quantum/MP.jl")
export MatrixProduct, MPS, MPO

# reexports from LinearAlgebra
export norm, normalize!

# reexports from EinExprs
export einexpr, inds

end

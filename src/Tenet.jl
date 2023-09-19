module Tenet

include("Helpers.jl")

include("Tensor.jl")
export Tensor, contract, dim, expand

include("Numerics.jl")

include("TensorNetwork.jl")
export TensorNetwork, tensors, arrays, select, slice!
export Domain, domain
export contract, contract!

include("Transformations.jl")
export transform, transform!

include("Quantum/Quantum.jl")
export Quantum
export Boundary, boundary, Open, Periodic, Infinite
export Plug, plug, Property, State, Operator
export sites, fidelity

export MatrixProduct, MPS, MPO
export ProjectedEntangledPair, PEPS, PEPO

# reexports from LinearAlgebra
export norm, normalize!

# reexports from EinExprs
export einexpr, inds

end

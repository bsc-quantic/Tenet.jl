module Operations

using ..Tenet: Tenet, Tensor, vinds, inds, contract
using CUDA: CUDA

# inspired by Oceananigans.jl
abstract type Vendor end
struct Intel <: Vendor end
struct NVIDIA <: Vendor end
struct AMD <: Vendor end

abstract type Architecture end
struct CPU <: Architecture end
struct GPU{V<:Vendor} <: Architecture end

arch(tensor::Tensor) = arch(parent(tensor))
arch(::Array) = CPU()
arch(::CUDA.CuArray) = GPU{NVIDIA}()

# TODO move `contract` to Operations

include("SVD.jl")
include("SimpleUpdate.jl")

end # module

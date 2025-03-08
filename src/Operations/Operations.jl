module Operations

using ..Tenet: Tenet, Tensor, inds
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

include("SVD.jl")
include("SimpleUpdate.jl")

end # module

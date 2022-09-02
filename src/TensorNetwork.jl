using WhereTraits
using OptimizedEinsum
using NamedDims

abstract type AbstractTensorNetwork end

ntensors(_::AbstractTensorNetwork) = error("No implementation found")
ninds(_::AbstractTensorNetwork) = error("No implementation found")

struct TensorNetwork <: AbstractTensorNetwork
    tensor_map::Dict{Int,NamedDimsArray}
    ind_map::Dict{Symbol,Vector{Int}}
end

inds(tn::TensorNetwork) = keys(tn.ind_map)
tensors(tn::TensorNetwork) = values(tn.tensor_map)
hyperinds(tn::TensorNetwork) = error("not implemented yet")

ntensors(tn::TensorNetwork) = length(tn.tensor_map)
ninds(tn::TensorNetwork) = length(tn.ind_map)

@traits ntensors(x::T) where {T<:AbstractTensorNetwork,hasfield(T, :tn)} = ntensors(x.tn)
@traits ninds(x::T) where {T<:AbstractTensorNetwork,hasfield(T, :tn)} = ninds(x.tn)
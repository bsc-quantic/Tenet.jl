module TenetChainRulesCoreExt

using Tenet
using ChainRulesCore

function ChainRulesCore.ProjectTo(tensor::T) where {T<:Tensor}
    ProjectTo{T}(; data = ProjectTo(tensor.data), inds = tensor.inds)
end

function (projector::ProjectTo{T})(dx::Union{T,Tangent{T}}) where {T<:Tensor}
    T(projector.data(dx.data), projector.inds)
end

ChainRulesCore.frule((_, Δ, _), T::Type{<:Tensor}, data, inds) = T(data, inds), T(Δ, inds)

Tensor_pullback(Δ) = (NoTangent(), Δ.data, NoTangent())
Tensor_pullback(Δ::AbstractThunk) = Tensor_pullback(unthunk(Δ))
ChainRulesCore.rrule(T::Type{<:Tensor}, data, inds) = T(data, inds), Tensor_pullback

# NOTE fix problem with vector generator in `contract`
@non_differentiable Tenet.__omeinsum_sym2str(x)

# WARN type-piracy
@non_differentiable setdiff(s::Base.AbstractVecOrTuple{Symbol}, itrs::Base.AbstractVecOrTuple{Symbol}...)
@non_differentiable union(s::Base.AbstractVecOrTuple{Symbol}, itrs::Base.AbstractVecOrTuple{Symbol}...)
@non_differentiable intersect(s::Base.AbstractVecOrTuple{Symbol}, itrs::Base.AbstractVecOrTuple{Symbol}...)
@non_differentiable symdiff(s::Base.AbstractVecOrTuple{Symbol}, itrs::Base.AbstractVecOrTuple{Symbol}...)

function ChainRulesCore.ProjectTo(tn::TensorNetwork)
    ProjectTo{TensorNetwork}(; tensors = ProjectTo(tensors(tn)))
end

function (projector::ProjectTo{TensorNetwork})(dx::TensorNetwork)
    Tangent{TensorNetwork}(tensormap = projector.tensors(tensors(dx)), indexmap = NoTangent())
end

function (projector::ProjectTo{TensorNetwork})(dx::Tangent{TensorNetwork})
    dx.tensormap isa NoTangent && return NoTangent()
    Tangent{TensorNetwork}(tensormap = projector.tensors(dx.tensors), indexmap = NoTangent())
end

function Base.:+(x::TensorNetwork, Δ::Tangent{TensorNetwork})
    # TODO match tensors by indices
    tensors = map(+, tensors(x), Δ.tensormap)

    # TODO create function fitted for this? or maybe standardize constructors?
    TensorNetwork(tensors)
end

function ChainRulesCore.frule((_, Δ), ::Type{TensorNetwork}, tensors)
    TensorNetwork(tensors), Tangent{TensorNetwork}(tensormap = Δ, indexmap = NoTangent())
end

TensorNetwork_pullback(Δ::Tangent{TensorNetwork}) = (NoTangent(), Δ.tensormap)
TensorNetwork_pullback(Δ::AbstractThunk) = TensorNetwork_pullback(unthunk(Δ))
function ChainRulesCore.rrule(::Type{TensorNetwork}, tensors)
    TensorNetwork(tensors), TensorNetwork_pullback
end

end

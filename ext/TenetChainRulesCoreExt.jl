module TenetChainRulesCoreExt

using Tenet
using ChainRulesCore

function ChainRulesCore.ProjectTo(tensor::T) where {T<:Tensor}
    ProjectTo{T}(; data = ProjectTo(tensor.data), inds = tensor.inds, meta = tensor.meta)
end

function (projector::ProjectTo{T})(dx::Union{T,Tangent{T}}) where {T<:Tensor}
    T(projector.data(dx.data), projector.inds; projector.meta...)
end

ChainRulesCore.frule((_, Δ, _), T::Type{<:Tensor}, data, inds; meta...) = T(data, inds; meta...), T(Δ, inds; meta...)

Tensor_pullback(Δ) = (NoTangent(), Δ.data, NoTangent())
Tensor_pullback(Δ::AbstractThunk) = Tensor_pullback(unthunk(Δ))
ChainRulesCore.rrule(T::Type{<:Tensor}, data, inds; meta...) = T(data, inds; meta...), Tensor_pullback

@non_differentiable copy(tn::TensorNetwork)

# NOTE fix problem with vector generator in `contract`
@non_differentiable Tenet.__omeinsum_sym2str(x)

# WARN type-piracy
@non_differentiable setdiff(s::Base.AbstractVecOrTuple{Symbol}, itrs::Base.AbstractVecOrTuple{Symbol}...)
@non_differentiable union(s::Base.AbstractVecOrTuple{Symbol}, itrs::Base.AbstractVecOrTuple{Symbol}...)
@non_differentiable intersect(s::Base.AbstractVecOrTuple{Symbol}, itrs::Base.AbstractVecOrTuple{Symbol}...)
@non_differentiable symdiff(s::Base.AbstractVecOrTuple{Symbol}, itrs::Base.AbstractVecOrTuple{Symbol}...)

function ChainRulesCore.ProjectTo(tn::T) where {T<:TensorNetwork}
    ProjectTo{T}(; tensors = ProjectTo(tn.tensors), metadata = tn.metadata)
end

function (projector::ProjectTo{T})(dx::Union{T,Tangent{T}}) where {A<:Ansatz,T<:TensorNetwork{A}}
    TensorNetwork{A}(projector.tensors(dx.tensors); projector.metadata...)
end

function Base.:+(x::TensorNetwork{A}, Δ::Tangent{TensorNetwork}) where {A<:Ansatz}
    tensors = map(+, x.tensors, Δ.tensors)
    TensorNetwork{A}(tensors; x.metadata...)
end

function ChainRulesCore.frule((_, Δ), T::Type{<:TensorNetwork}, tensors; metadata...)
    T(tensors; metadata...), Tangent{TensorNetwork}(tensors = Δ)
end

TensorNetwork_pullback(Δ::Tangent{TensorNetwork}) = (NoTangent(), Δ.tensors)
TensorNetwork_pullback(Δ::AbstractThunk) = TensorNetwork_pullback(unthunk(Δ))
function ChainRulesCore.rrule(T::Type{<:TensorNetwork}, tensors; metadata...)
    T(tensors; metadata...), TensorNetwork_pullback
end

end
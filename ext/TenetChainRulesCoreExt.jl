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

_Tensor_pullback(Δ) = (NoTangent(), Δ.data, NoTangent())
_Tensor_pullback(Δ::AbstractThunk) = _Tensor_pullback(unthunk(Δ))
ChainRulesCore.rrule(T::Type{<:Tensor}, data, inds; meta...) = T(data, inds; meta...), _Tensor_pullback

@non_differentiable copy(tn::TensorNetwork)

# NOTE fix problem with vector generator in `contract`
@non_differentiable Tenet.__omeinsum_sym2str(x)

# WARN type-piracy
@non_differentiable setdiff(s::Base.AbstractVecOrTuple{Symbol}, itrs::Base.AbstractVecOrTuple{Symbol}...)
@non_differentiable union(s::Base.AbstractVecOrTuple{Symbol}, itrs::Base.AbstractVecOrTuple{Symbol}...)
@non_differentiable intersect(s::Base.AbstractVecOrTuple{Symbol}, itrs::Base.AbstractVecOrTuple{Symbol}...)
@non_differentiable symdiff(s::Base.AbstractVecOrTuple{Symbol}, itrs::Base.AbstractVecOrTuple{Symbol}...)

end
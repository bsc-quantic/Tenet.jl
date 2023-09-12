module TenetChainRulesCoreExt

using Tenet
using ChainRulesCore

ChainRulesCore.ProjectTo(tensor::T) where {T<:Tensor} =
    ProjectTo{T}(; data = ProjectTo(tensor.data), inds = tensor.inds, meta = tensor.meta)

(projector::ProjectTo{T})(dx::Union{T,Tangent{T}}) where {T<:Tensor} =
    T(projector.data(dx.data), projector.inds; projector.meta...)

ChainRulesCore.frule((_, Δ, _), T::Type{<:Tensor}, data, inds; meta...) = T(data, inds; meta...), T(Δ, inds; meta...)

function ChainRulesCore.rrule(T::Type{<:Tensor}, data, inds; meta...)
    Tensor_pullback(Δ) = (NoTangent(), Δ.data, NoTangent())
    return T(data, inds; meta...), Tensor_pullback
end

@non_differentiable copy(tn::TensorNetwork)

# NOTE fix problem with vector generator in `contract`
@non_differentiable Tenet.__omeinsum_sym2str(x)

# WARN type-piracy
@non_differentiable setdiff(s::Base.AbstractVecOrTuple{Symbol}, itrs::Base.AbstractVecOrTuple{Symbol}...)
@non_differentiable union(s::Base.AbstractVecOrTuple{Symbol}, itrs::Base.AbstractVecOrTuple{Symbol}...)
@non_differentiable intersect(s::Base.AbstractVecOrTuple{Symbol}, itrs::Base.AbstractVecOrTuple{Symbol}...)
@non_differentiable symdiff(s::Base.AbstractVecOrTuple{Symbol}, itrs::Base.AbstractVecOrTuple{Symbol}...)

end
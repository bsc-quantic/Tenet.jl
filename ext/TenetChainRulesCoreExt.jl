module TenetChainRulesCoreExt

using Tenet
using ChainRulesCore

ChainRulesCore.ProjectTo(tensor::T) where {T<:Tensor} =
    ProjectTo{T}(; data = ProjectTo(tensor.data), inds = tensor.inds, meta = tensor.meta)

(projector::ProjectTo{T})(dx::Union{T,Tangent{T}}) where {T<:Tensor} =
    T(projector.data(dx.data), projector.inds; projector.meta...)

function ChainRulesCore.rrule(::Type{Tensor{T,N,A}}, data, inds; meta...) where {T,N,A}
    return Tensor(data, inds; meta...), function Tensor_pullback(_)
        (NoTangent(), data, NoTangent())
    end
end

function ChainRulesCore.rrule(T::Type{<:Tensor}, data, inds; meta...)
    Tensor_pullback(Δ) = (NoTangent(), Δ.data, NoTangent())
    return T(data, inds; meta...), Tensor_pullback
end

# WARN type-piracy
@non_differentiable setdiff(s::Base.AbstractVecOrTuple{Symbol}, itrs::Base.AbstractVecOrTuple{Symbol}...)
@non_differentiable union(s::Base.AbstractVecOrTuple{Symbol}, itrs::Base.AbstractVecOrTuple{Symbol}...)
@non_differentiable intersect(s::Base.AbstractVecOrTuple{Symbol}, itrs::Base.AbstractVecOrTuple{Symbol}...)
@non_differentiable symdiff(s::Base.AbstractVecOrTuple{Symbol}, itrs::Base.AbstractVecOrTuple{Symbol}...)

end
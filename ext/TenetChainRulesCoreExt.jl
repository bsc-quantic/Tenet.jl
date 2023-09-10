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
function ChainRulesCore.rrule(::typeof(setdiff), s, itrs...)
    setdiff_pullback(_) = fill(NoTangent(), 2 + length(itrs))
    return setdiff(s, itrs...), setdiff_pullback
end

function ChainRulesCore.rrule(::typeof(union), s, itrs...)
    union_pullback(_) = fill(NoTangent(), 2 + length(itrs))
    return union(s, itrs...), union_pullback
end

function ChainRulesCore.rrule(::typeof(intersect), s, itrs...)
    intersect_pullback(_) = fill(NoTangent(), 2 + length(itrs))
    return intersect(s, itrs...), intersect_pullback
end

end
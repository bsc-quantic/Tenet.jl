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

end
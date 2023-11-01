module TenetChainRulesCoreExt

using Tenet
using Classes
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

function ChainRulesCore.ProjectTo(tn::T) where {T<:absclass(TensorNetwork)}
    # TODO create function to extract extra fields
    fields = map(fieldnames(T)) do fieldname
        if fieldname === :tensors
            :tensors => ProjectTo(tn.tensors)
        else
            fieldname => getfield(tn, fieldname)
        end
    end
    ProjectTo{T}(; fields...)
end

function (projector::ProjectTo{T})(dx::Union{T,Tangent{T}}) where {T<:absclass(TensorNetwork)}
    dx.tensors isa NoTangent && return NoTangent()
    Tangent{TensorNetwork}(tensors = projector.tensors(dx.tensors))
end

function Base.:+(x::T, Δ::Tangent{TensorNetwork}) where {T<:absclass(TensorNetwork)}
    # TODO match tensors by indices
    tensors = map(+, tensors(x), Δ.tensors)

    # TODO create function fitted for this? or maybe standardize constructors?
    T(map(fieldnames(T)) do fieldname
        if fieldname === :tensors
            tensors
        else
            getfield(x, fieldname)
        end
    end...)
end

function ChainRulesCore.frule((_, Δ), T::Type{<:absclass(TensorNetwork)}, tensors)
    T(tensors), Tangent{TensorNetwork}(tensors = Δ)
end

TensorNetwork_pullback(Δ::Tangent{TensorNetwork}) = (NoTangent(), Δ.tensors)
TensorNetwork_pullback(Δ::AbstractThunk) = TensorNetwork_pullback(unthunk(Δ))
function ChainRulesCore.rrule(T::Type{<:absclass(TensorNetwork)}, tensors)
    T(tensors), TensorNetwork_pullback
end

end

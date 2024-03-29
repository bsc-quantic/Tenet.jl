# `Tensor` constructor
Tensor_pullback(Δ) = (NoTangent(), Δ.data, NoTangent())
Tensor_pullback(Δ::AbstractThunk) = Tensor_pullback(unthunk(Δ))
ChainRulesCore.rrule(T::Type{<:Tensor}, data, inds) = T(data, inds), Tensor_pullback

# `TensorNetwork` constructor
TensorNetwork_pullback(Δ::Tangent{TensorNetwork}) = (NoTangent(), Δ.tensormap)
TensorNetwork_pullback(Δ::AbstractThunk) = TensorNetwork_pullback(unthunk(Δ))
function ChainRulesCore.rrule(::Type{TensorNetwork}, tensors)
    TensorNetwork(tensors), TensorNetwork_pullback
end

# `Base.conj` methods
conj_pullback(Δ::Tensor) = (NoTangent(), conj(Δ))
conj_pullback(Δ::Tangent{Tensor}) = (NoTangent(), conj(Δ))
conj_pullback(Δ::Tangent{TensorNetwork}) = (NoTangent(), conj(Δ))
conj_pullback(Δ::AbstractThunk) = conj_pullback(unthunk(Δ))

function ChainRulesCore.rrule(::typeof(Base.conj), tn::Tensor)
    conj(tn), conj_pullback
end

function ChainRulesCore.rrule(::typeof(Base.conj), tn::TensorNetwork)
    conj(tn), conj_pullback
end

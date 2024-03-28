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

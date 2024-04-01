# `Tensor` constructor
ChainRulesCore.frule((_, Δ, _), T::Type{<:Tensor}, data, inds) = T(data, inds), T(Δ, inds)

# `TensorNetwork` constructor
ChainRulesCore.frule((_, Δ), ::Type{TensorNetwork}, tensors) = TensorNetwork(tensors), TensorNetworkTangent(Δ)

# `Base.conj` methods
ChainRulesCore.frule((_, Δ), ::typeof(Base.conj), tn::Tensor) = conj(tn), conj(Δ)

ChainRulesCore.frule((_, Δ), ::typeof(Base.conj), tn::TensorNetwork) = conj(tn), conj(Δ)

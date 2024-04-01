# `Tensor` constructor
ChainRulesCore.frule((_, Δ, _), T::Type{<:Tensor}, data, inds) = T(data, inds), T(Δ, inds)

# `TensorNetwork` constructor
ChainRulesCore.frule((_, Δ), ::Type{TensorNetwork}, tensors) = TensorNetwork(tensors), Tangent{TensorNetwork}(Δ)

# `Base.conj` methods
ChainRulesCore.frule((_, Δ), ::typeof(Base.conj), tn::Tensor) = conj(tn), conj(Δ)

ChainRulesCore.frule((_, Δ), ::typeof(Base.conj), tn::TensorNetwork) = conj(tn), conj(Δ)

# `Base.merge` methods
ChainRulesCore.frule((_, ȧ, ḃ), ::typeof(Base.merge), a::TensorNetwork, b::TensorNetwork) = merge(a, b), merge(ȧ, ḃ)

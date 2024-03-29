# `Tensor` constructor
ChainRulesCore.frule((_, Δ, _), T::Type{<:Tensor}, data, inds) = T(data, inds), T(Δ, inds)

# `TensorNetwork` constructor
function ChainRulesCore.frule((_, Δ), ::Type{TensorNetwork}, tensors)
    TensorNetwork(tensors), Tangent{TensorNetwork}(tensormap = Δ, indexmap = NoTangent())
end

# `Base.conj` methods
function ChainRulesCore.frule((_, Δ), ::typeof(Base.conj), tn::Tensor)
    conj(tn), conj(Δ)
end

function ChainRulesCore.frule((_, Δ), ::typeof(Base.conj), tn::TensorNetwork)
    conj(tn), conj(Δ)
end

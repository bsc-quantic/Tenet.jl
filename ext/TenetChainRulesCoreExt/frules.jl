# `Tensor` constructor
ChainRulesCore.frule((_, Δ, _), T::Type{<:Tensor}, data, inds) = T(data, inds), T(Δ, inds)

# `TensorNetwork` constructor
function ChainRulesCore.frule((_, Δ), ::Type{TensorNetwork}, tensors)
    TensorNetwork(tensors), Tangent{TensorNetwork}(tensormap = Δ, indexmap = NoTangent())
end

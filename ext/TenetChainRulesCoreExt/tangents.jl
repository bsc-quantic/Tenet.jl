function Base.:+(x::TensorNetwork, Δ::Tangent{TensorNetwork})
    # TODO match tensors by indices
    tensors = map(+, tensors(x), Δ.tensormap)

    # TODO create function fitted for this? or maybe standardize constructors?
    TensorNetwork(tensors)
end

function Base.conj(Δ::Tangent{<:Tensor})
    Tangent{Tensor}(data = conj(Δ.data), inds = NoTangent())
end

function Base.conj(Δ::Tangent{TensorNetwork})
    Tangent{TensorNetwork}(tensormap = Tensor[conj(t) for t in Δ.tensormap], indexmap = NoTangent())
end

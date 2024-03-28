function Base.:+(x::TensorNetwork, Δ::Tangent{TensorNetwork})
    # TODO match tensors by indices
    tensors = map(+, tensors(x), Δ.tensormap)

    # TODO create function fitted for this? or maybe standardize constructors?
    TensorNetwork(tensors)
end

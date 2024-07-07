# `Tensor` constructor
ChainRulesCore.frule((_, Δ, _), T::Type{<:Tensor}, data, inds) = T(data, inds), T(Δ, inds)

# `TensorNetwork` constructor
ChainRulesCore.frule((_, Δ), ::Type{TensorNetwork}, tensors) = TensorNetwork(tensors), TensorNetworkTangent(Δ)

# `Base.conj` methods
ChainRulesCore.frule((_, Δ), ::typeof(Base.conj), tn::Tensor) = conj(tn), conj(Δ)

ChainRulesCore.frule((_, Δ), ::typeof(Base.conj), tn::TensorNetwork) = conj(tn), conj(Δ)

# `Base.merge` methods
ChainRulesCore.frule((_, ȧ, ḃ), ::typeof(Base.merge), a::TensorNetwork, b::TensorNetwork) = merge(a, b), merge(ȧ, ḃ)

# `contract` methods
function ChainRulesCore.frule((_, ẋ), ::typeof(contract), x::Tensor; kwargs...)
    return contract(x; kwargs...), contract(ẋ; kwargs...)
end

function ChainRulesCore.frule((_, ȧ, ḃ), ::typeof(contract), a::Tensor, b::Tensor; kwargs...)
    c = contract(a, b; kwargs...)
    ċ = contract(ȧ, b; kwargs...) + contract(a, ḃ; kwargs...)
    return c, ċ
end

function ChainRulesCore.frule((_, ẋ, _), ::Type{Quantum}, x::TensorNetwork, sites)
    y = Quantum(x, sites)
    ẏ = Tangent{Quantum}(; tn=ẋ)
    return y, ẏ
end

ChainRulesCore.frule((_, ẋ), ::Type{T}, x::Quantum) where {T<:Ansatz} = T(x), Tangent{T}(; super=ẋ)

function ChainRulesCore.frule((_, ẋ, _), ::Type{T}, x::Quantum, boundary) where {T<:Ansatz}
    return T(x, boundary), Tangent{T}(; super=ẋ, boundary=NoTangent())
end

using Tenet: AbstractTensorNetwork, AbstractQuantum

# `Tensor` constructor
ChainRulesCore.frule((_, Δ, _), T::Type{<:Tensor}, data, inds) = T(data, inds), T(Δ, inds)

# `TensorNetwork` constructor
ChainRulesCore.frule((_, Δ), ::Type{TensorNetwork}, tensors) = TensorNetwork(tensors), TensorNetworkTangent(Δ)

# `Quantum` constructor
function ChainRulesCore.frule((_, ẋ, _), ::Type{Quantum}, x::TensorNetwork, sites)
    return Quantum(x, sites), Tangent{Quantum}(; tn=ẋ, sites=NoTangent())
end

# `Ansatz` constructor
function ChainRulesCore.frule((_, ẋ), ::Type{Ansatz}, x::Quantum, lattice)
    return Ansatz(x, lattice), Tangent{Ansatz}(; tn=ẋ, lattice=NoTangent())
end

# `Base.conj` methods
ChainRulesCore.frule((_, Δ), ::typeof(Base.conj), tn::Tensor) = conj(tn), conj(Δ)

ChainRulesCore.frule((_, Δ), ::typeof(Base.conj), tn::AbstractTensorNetwork) = conj(tn), conj(Δ)

# `Base.merge` methods
function ChainRulesCore.frule((_, ȧ, ḃ), ::typeof(Base.merge), a::AbstractTensorNetwork, b::AbstractTensorNetwork)
    return merge(a, b), merge(ȧ, ḃ)
end

# `contract` methods
function ChainRulesCore.frule((_, ẋ), ::typeof(contract), x::Tensor; kwargs...)
    return contract(x; kwargs...), contract(ẋ; kwargs...)
end

function ChainRulesCore.frule((_, ȧ, ḃ), ::typeof(contract), a::Tensor, b::Tensor; kwargs...)
    c = contract(a, b; kwargs...)
    ċ = contract(ȧ, b; kwargs...) + contract(a, ḃ; kwargs...)
    return c, ċ
end

using ChainRulesCore: AbstractTangent

struct TensorNetworkTangent <: AbstractTangent
    tensors::Dict{Vector{Symbol},Tensor} # `Tensor` is its own `Tangent` type
end

function TensorNetworkTangent(tensors::Vector{<:Tensor})
    TensorNetworkTangent(Dict(inds(tensor) => tensor for tensor in tensors))
end

Tenet.tensors(tn::TensorNetworkTangent) = collect(values(tn.tensors))

# additive identity for `TensorNetwork` tangent
Base.zero(::Type{<:TensorNetworkTangent}) = TensorNetworkTangent(Tensor[])

# gradient accumulation
function Base.:(+)(Δa::TensorNetworkTangent, Δb::TensorNetworkTangent)
    TensorNetworkTangent(mergewith(+, Δa.tensors, Δb.tensors))
end

# scalar multiplication
Base.:(*)(α::Number, Δ::TensorNetworkTangent) = TensorNetworkTangent(Dict(inds(t) => α * t for (inds, t) in tensors(Δ)))
Base.:(*)(Δ::TensorNetworkTangent, α::Number) = α * Δ

# primal-gradient addition
function Base.:(+)(primal::TensorNetwork, Δ::TensorNetworkTangent)
    @assert all(inds(t) in keys(Δ.tensors) for t in tensors(primal))
    # TODO iterate through `Δ`
    TensorNetwork(map(tensors(primal)) do t
        # TODO what if multiplicity > 1?
        Δt = get(Δ.tensors, inds(t), ZeroTangent())
        t + Δt
    end)
end

# iteration interface
Base.IteratorSize(::Type{TensorNetworkTangent}) = Base.HasLength()
Base.length(Δ::TensorNetworkTangent) = length(tensors(Δ))

Base.IteratorEltype(::Type{TensorNetworkTangent}) = Base.HasEltype()
Base.eltype(::Type{TensorNetworkTangent}) = Tensor
Base.eltype(::TensorNetworkTangent) = Tensor

Base.iterate(Δ::TensorNetworkTangent, state = 1) = iterate(values(Δ.tensors), state)

Base.merge(Δa::TensorNetworkTangent, Δb::TensorNetworkTangent) = Δa + Δb

Base.conj(Δ::Tangent{<:Tensor}) = Tangent{Tensor}(data = conj(Δ.data), inds = NoTangent())
Base.conj(Δ::TensorNetworkTangent) = TensorNetworkTangent(Dict(inds => conj(t) for (inds, t) in Δ.tensors))

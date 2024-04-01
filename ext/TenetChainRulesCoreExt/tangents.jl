using ChainRulesCore: AbstractTangent

function Tangent{TensorNetwork}(tensors::Vector{<:Tensor})
    Tangent{TensorNetwork}(Dict{Vector{Symbol},Tensor}(inds(tensor) => tensor for tensor in tensors))
end

function Tangent{TensorNetwork}(tensors::Dict{Vector{Symbol},Tensor})
    Tangent{TensorNetwork}(; tensormap = tensors, indexmap = NoTangent())
end

Tenet.tensors(tn::Tangent{TensorNetwork}) = collect(values(tn.tensormap))

# additive identity for `TensorNetwork` tangent
Base.zero(::Type{<:Tangent{TensorNetwork}}) = Tangent{TensorNetwork}(Tensor[])

# gradient accumulation
function Base.:(+)(Δa::Tangent{TensorNetwork}, Δb::Tangent{TensorNetwork})
    Tangent{TensorNetwork}(mergewith(+, Δa.tensormap, Δb.tensormap))
end

# scalar multiplication
Base.:(*)(α::Number, Δ::Tangent{TensorNetwork}) =
    Tangent{TensorNetwork}(Dict(inds(t) => α * t for (inds, t) in tensors(Δ)))
Base.:(*)(Δ::Tangent{TensorNetwork}, α::Number) = α * Δ

# primal-gradient addition
function Base.:(+)(primal::TensorNetwork, Δ::Tangent{TensorNetwork})
    @assert all(inds(t) in keys(Δ.tensormap) for t in tensors(primal))
    # TODO iterate through `Δ`
    TensorNetwork(map(tensors(primal)) do t
        # TODO what if multiplicity > 1?
        Δt = get(Δ.tensormap, inds(t), ZeroTangent())
        t + Δt
    end)
end

# iteration interface
Base.IteratorSize(::Type{Tangent{TensorNetwork}}) = Base.HasLength()
Base.length(Δ::Tangent{TensorNetwork}) = length(tensors(Δ))

Base.IteratorEltype(::Type{Tangent{TensorNetwork}}) = Base.HasEltype()
Base.eltype(::Type{Tangent{TensorNetwork}}) = Tensor
Base.eltype(::Tangent{TensorNetwork}) = Tensor

Base.iterate(Δ::Tangent{TensorNetwork}, state = 1) = iterate(values(Δ.tensormap), state)

Base.merge(Δa::Tangent{TensorNetwork}, Δb::Tangent{TensorNetwork}) = Δa + Δb

Base.conj(Δ::Tangent{<:Tensor}) = Tangent{Tensor}(data = conj(Δ.data), inds = NoTangent())
function Base.conj(Δ::Tangent{TensorNetwork})
    Tangent{TensorNetwork}(Dict{Vector{Symbol},Tensor}(inds => conj(t) for (inds, t) in Δ.tensormap))
end

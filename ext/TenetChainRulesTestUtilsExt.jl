__precompile__(false)

module TenetChainRulesTestUtilsExt

using Tenet
using ChainRulesCore
using ChainRulesTestUtils
using Random
using Graphs

const TensorNetworkTangent = Base.get_extension(Tenet, :TenetChainRulesCoreExt).TensorNetworkTangent

function ChainRulesTestUtils.rand_tangent(rng::AbstractRNG, tn::Vector{T}) where {T<:Tensor}
    if isempty(tn)
        return Vector{T}()
    else
        @invoke rand_tangent(rng::AbstractRNG, tn::AbstractArray)
    end
end

function ChainRulesTestUtils.rand_tangent(rng::AbstractRNG, x::TensorNetwork)
    return TensorNetworkTangent(Tensor[ProjectTo(tensor)(rand_tangent.(Ref(rng), tensor)) for tensor in tensors(x)])
end

function ChainRulesTestUtils.rand_tangent(rng::AbstractRNG, tn::Quantum)
    return Tangent{Quantum}(; tn=rand_tangent(rng, TensorNetwork(tn)), sites=NoTangent())
end

# WARN type-piracy,  used in `Quantum` constructor
ChainRulesTestUtils.rand_tangent(::AbstractRNG, tn::Dict{<:Site,Symbol}) = NoTangent()

function ChainRulesTestUtils.rand_tangent(rng::AbstractRNG, tn::Ansatz)
    return Tangent{Ansatz}(; tn=rand_tangent(rng, Quantum(tn)), lattice=NoTangent())
end

# WARN not really type-piracy but almost, used in `Ansatz` constructor
# ChainRulesTestUtils.rand_tangent(::AbstractRNG, tn::T) where {V,T<:MetaGraph{V,SimpleGraph{V},<:Site}} = NoTangent()

# WARN not really type-piracy but almost, used when testing `Ansatz`
# function ChainRulesTestUtils.test_approx(
#     actual::G, expected::G, msg; kwargs...
# ) where {G<:MetaGraph{Int64,SimpleGraph{Int64},<:Site}}
#     return actual == expected
# end

end

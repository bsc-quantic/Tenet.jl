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

ChainRulesTestUtils.rand_tangent(::AbstractRNG, lattice::Tenet.Lattice) = NoTangent()
ChainRulesTestUtils.test_approx(actual::Lattice, expected::Lattice, msg; kwargs...) = actual == expected

end

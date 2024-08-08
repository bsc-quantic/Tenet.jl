__precompile__(false)

module TenetChainRulesTestUtilsExt

using Tenet
using ChainRulesCore
using ChainRulesTestUtils
using Random

const TensorNetworkTangent = Base.get_extension(Tenet, :TenetChainRulesCoreExt).TensorNetworkTangent

function ChainRulesTestUtils.rand_tangent(rng::AbstractRNG, x::Vector{T}) where {T<:Tensor}
    if isempty(x)
        return Vector{T}()
    else
        @invoke rand_tangent(rng::AbstractRNG, x::AbstractArray)
    end
end

function ChainRulesTestUtils.rand_tangent(rng::AbstractRNG, x::TensorNetwork)
    return TensorNetworkTangent(Tensor[ProjectTo(tensor)(rand_tangent.(Ref(rng), tensor)) for tensor in tensors(x)])
end

function ChainRulesTestUtils.rand_tangent(rng::AbstractRNG, x::Quantum)
    return Tangent{Quantum}(; tn=rand_tangent(rng, TensorNetwork(x)), sites=NoTangent())
end

# WARN type-piracy
# NOTE used in `Quantum` constructor
ChainRulesTestUtils.rand_tangent(::AbstractRNG, x::Dict{<:Site,Symbol}) = NoTangent()

end

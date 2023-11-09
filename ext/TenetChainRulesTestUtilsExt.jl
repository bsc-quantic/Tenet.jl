module TenetChainRulesTestUtilsExt

using Tenet
using Tenet: AbstractTensorNetwork
using ChainRulesCore
using ChainRulesTestUtils
using Random

function ChainRulesTestUtils.rand_tangent(rng::AbstractRNG, x::T) where {T<:AbstractTensorNetwork}
    return Tangent{T}(
        tensormap = Tensor[ProjectTo(tensor)(rand_tangent.(Ref(rng), tensor)) for tensor in tensors(x)],
        indexmap = NoTangent(),
    )
end

end

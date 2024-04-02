module TenetChainRulesTestUtilsExt

using Tenet
using ChainRulesCore
using ChainRulesTestUtils
using Random

const TensorNetworkTangent = Base.get_extension(Tenet, :TenetChainRulesCoreExt).TensorNetworkTangent

function ChainRulesTestUtils.rand_tangent(rng::AbstractRNG, x::TensorNetwork)
    return TensorNetworkTangent(Tensor[ProjectTo(tensor)(rand_tangent.(Ref(rng), tensor)) for tensor in tensors(x)])
end

end

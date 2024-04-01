module TenetChainRulesTestUtilsExt

using Tenet
using ChainRulesCore
using ChainRulesTestUtils
using Random

function ChainRulesTestUtils.rand_tangent(rng::AbstractRNG, x::TensorNetwork)
    return Tangent{TensorNetwork}(Tensor[ProjectTo(tensor)(rand_tangent.(Ref(rng), tensor)) for tensor in tensors(x)])
end

end

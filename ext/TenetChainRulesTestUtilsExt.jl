module TenetChainRulesTestUtilsExt

using Tenet
using ChainRulesCore
using ChainRulesTestUtils
using Random
using Classes

function ChainRulesTestUtils.rand_tangent(rng::AbstractRNG, x::T) where {T<:absclass(TensorNetwork)}
    return Tangent{T}(tensors = Tensor[ProjectTo(tensor)(rand_tangent.(Ref(rng), tensor)) for tensor in tensors(x)])
end

end
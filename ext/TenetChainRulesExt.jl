module TenetChainRulesExt

using Tenet
using ChainRules

function (projector::ChainRules.ProjectTo{T})(dx::ChainRules.OneElement) where {T<:Tensor}
    return T(projector.data(dx.val), projector.inds)
end

end

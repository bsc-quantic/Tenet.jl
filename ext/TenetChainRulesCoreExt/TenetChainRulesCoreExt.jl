module TenetChainRulesCoreExt

using Tenet
using ChainRulesCore

include("projectors.jl")
include("non_diff.jl")
include("tangents.jl")
include("frules.jl")
include("rrules.jl")

end

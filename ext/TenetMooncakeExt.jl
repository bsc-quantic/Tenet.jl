module TenetMooncakeExt

using Tenet
using Mooncake: Mooncake, @from_rrule, DefaultCtx

@from_rrule DefaultCtx Tuple{typeof(contract),Tensor}
@from_rrule DefaultCtx Tuple{typeof(contract),Tensor,Tensor}

# function Mooncake.tangent(fdata::Mooncake.FData{<:NamedTuple{(:data, :inds)}}, tensor::Tensor)
#     @show fdata
#     @show tensor
# end

end # module

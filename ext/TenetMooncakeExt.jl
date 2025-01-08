module TenetMooncakeExt

using Tenet
using Mooncake: Mooncake, @from_rrule, DefaultCtx

Mooncake.to_cr_tangent(tensor::Tensor) = tensor

@from_rrule DefaultCtx Tuple{typeof(contract),Tensor} true
@from_rrule DefaultCtx Tuple{typeof(contract),Tensor,Tensor} true

# function Mooncake.tangent(fdata::Mooncake.FData{<:NamedTuple{(:data, :inds)}}, tensor::Tensor)
#     @show fdata
#     @show tensor
# end

end # module

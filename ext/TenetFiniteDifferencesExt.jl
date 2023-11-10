module TenetFiniteDifferencesExt

using Tenet
using Tenet: AbstractTensorNetwork
using FiniteDifferences

function FiniteDifferences.to_vec(x::T) where {T<:AbstractTensorNetwork}
    x_vec, back = to_vec(tensors(x))
    TensorNetwork_from_vec(v) = T(back(v))

    return x_vec, TensorNetwork_from_vec
end

end

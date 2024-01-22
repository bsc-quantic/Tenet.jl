module TenetFiniteDifferencesExt

using Tenet
using FiniteDifferences

function FiniteDifferences.to_vec(x::TensorNetwork)
    x_vec, back = to_vec(tensors(x))
    TensorNetwork_from_vec(v) = TensorNetwork(back(v))

    return x_vec, TensorNetwork_from_vec
end

end

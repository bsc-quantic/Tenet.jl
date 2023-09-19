module TenetFiniteDifferencesExt

using Tenet
using FiniteDifferences

function FiniteDifferences.to_vec(x::TensorNetwork{D}) where {D}
    x_vec, back = to_vec(x.tensors)
    function TensorNetwork_from_vec(v)
        tensors = back(v)
        TensorNetwork{D}(tensors; x.metadata...)
    end

    return x_vec, TensorNetwork_from_vec
end

end
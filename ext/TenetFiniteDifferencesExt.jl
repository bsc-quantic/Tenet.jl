module TenetFiniteDifferencesExt

using Tenet
using FiniteDifferences

function FiniteDifferences.to_vec(x::TensorNetwork{A}) where {A<:Ansatz}
    x_vec, back = to_vec(x.tensors)
    function TensorNetwork_from_vec(v)
        tensors = back(v)
        TensorNetwork{A}(tensors; x.metadata...)
    end

    return x_vec, TensorNetwork_from_vec
end

end

module TenetFiniteDifferencesExt

using Tenet
using FiniteDifferences

function FiniteDifferences.to_vec(x::TensorNetwork)
    x_vec, back = to_vec(tensors(x))
    TensorNetwork_from_vec(v) = TensorNetwork(back(v))

    return x_vec, TensorNetwork_from_vec
end

# fixes wrong tangents on `TensorNetworkTangent` objects
# see: "different ndims" test in `ChainRules_test.jl`
function FiniteDifferences.to_vec(x::Dict{Vector{Symbol},Tensor})
    tensors = sort!(collect(values(x)); by=inds)
    x_vec, back = to_vec(tensors)
    Dict_from_vec(v) = Dict(zip(inds.(tensors), back(v)))

    return x_vec, Dict_from_vec
end

end

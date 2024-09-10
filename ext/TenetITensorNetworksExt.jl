module TenetITensorNetworksExt

using Tenet
using ITensorNetworks: ITensorNetworks, ITensorNetwork, ITensor, siteinds, plev, vertices
const ITensors = ITensorNetworks.ITensors
const DataGraphs = ITensorNetworks.DataGraphs

Base.convert(::Type{TensorNetwork}, tn::ITensorNetwork) = TensorNetwork([convert(Tensor, tn[v]) for v in vertices(tn)])

function Base.convert(::Type{Quantum}, tn::ITensorNetwork)
    sitedict = Dict(
        map(pairs(DataGraphs.vertex_data(siteinds(tn)))) do (loc, index)
            index = only(index)
            primelevel = plev(index)
            @assert primelevel ∈ (0, 1)
            Site(loc; dual=Bool(primelevel)) => Symbol(ITensors.id(index))
        end,
    )
    return Quantum(convert(TensorNetwork, tn), sitedict)
end

end

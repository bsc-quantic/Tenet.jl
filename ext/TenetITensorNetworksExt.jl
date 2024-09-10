module TenetITensorNetworksExt

using Tenet
using ITensorNetworks: ITensorNetworks, ITensorNetwork, ITensor, siteinds, plev, vertices
const ITensors = ITensorNetworks.ITensors
const DataGraphs = ITensorNetworks.DataGraphs

Tenet.TensorNetwork(tn::ITensorNetwork) = TensorNetwork([tn[v] for v in vertices(tn)])

function Tenet.Quantum(tn::ITensorNetwork)
    sitedict = Dict(
        map(pairs(DataGraphs.vertex_data(siteinds(tn)))) do (loc, index)
            index = only(index)
            primelevel = plev(index)
            @assert primelevel ∈ (0, 1)
            Site(loc; dual=Bool(primelevel)) => Symbol(ITensors.id(index))
        end,
    )
    return Quantum(TensorNetwork(tn), sitedict)
end

end

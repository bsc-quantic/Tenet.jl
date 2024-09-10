module TenetITensorNetworksExt

using Tenet
using ITensorNetworks: ITensorNetworks, ITensorNetwork, ITensor, Index, siteinds, plev, vertices, rename_vertices
const ITensors = ITensorNetworks.ITensors
const DataGraphs = ITensorNetworks.DataGraphs
const TenetITensorsExt = Base.get_extension(Tenet, :TenetITensorsExt)

Base.convert(::Type{TensorNetwork}, tn::ITensorNetwork) = TensorNetwork([convert(Tensor, tn[v]) for v in vertices(tn)])

function Base.convert(::Type{ITensorNetwork}, tn::Tenet.AbstractTensorNetwork; inds=Dict{Symbol,Index}())
    return ITensorNetwork(convert(Vector{ITensor}, tn; inds))
end

function Base.convert(::Type{Quantum}, tn::ITensorNetwork)
    sitedict = Dict(
        map(pairs(DataGraphs.vertex_data(siteinds(tn)))) do (loc, index)
            index = only(index)
            primelevel = plev(index)
            @assert primelevel ∈ (0, 1)

            # NOTE ITensors' Index's tag only has space for 16 characters
            tag = ITensors.id(index)
            Site(loc; dual=Bool(primelevel)) => TenetITensorsExt.symbolize(index)
        end,
    )
    return Quantum(convert(TensorNetwork, tn), sitedict)
end

function Base.convert(::Type{ITensorNetwork}, tn::Tenet.AbstractQuantum)
    itn = @invoke convert(ITensorNetwork, tn::Tenet.AbstractTensorNetwork)

    return rename_vertices(itn) do v
        itensor = itn[v]
        indices = map(x -> Symbol(replace(x, "\"" => "")), string.(ITensors.tags.(ITensors.inds(itensor))))
        tensor = only(tensors(tn; contains=indices))
        physical_index = only(inds(tn; set=:physical) ∩ inds(tensor))
        return sites(tn; at=physical_index).id
    end
end

end

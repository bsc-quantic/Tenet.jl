module TenetITensorNetworksExt

using Tenet
using ITensorNetworks: ITensorNetworks, ITensorNetwork, IndsNetwork, plev, rename_vertices
using ITensors: ITensors, ITensor, Index, siteinds
using Graphs: vertices
using DataGraphs: DataGraphs, underlying_graph
using NamedGraphs: NamedGraph, position_graph
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
        return Tenet.id(sites(tn; at=physical_index))
    end
end

function Base.convert(::Type{Lattice}, s::IndsNetwork)
    # NOTE they don't suffer from the issue of removing a vertex from SimpleGraph (i.e. vertex renaming), because their vertice list keeps ordered
    ng = underlying_graph(s) # namedgraph
    g = position_graph(ng) # simplegraph
    lanes = Lane.(collect(vertices(ng)))
    return Lattice(lanes, copy(g))
end

function Base.convert(::Type{IndsNetwork}, l::Lattice)
    return NamedGraph(copy(l.graph), Tenet.id.(vertices(l)))
end

function Base.convert(::Type{Ansatz}, tn::ITensorNetwork)
    return Ansatz(convert(Quantum, tn), convert(Lattice, siteinds(tn)))
end

function Base.convert(::Type{ITensorNetwork}, tn::Tenet.AbstractAnsatz)
    return @invoke convert(ITensorNetwork, tn::Tenet.AbstractQuantum)
end

end

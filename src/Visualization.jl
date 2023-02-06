using Graphs: SimpleGraph, Edge, nv
using GraphMakie: graphplot!, GraphPlot
using Combinatorics: combinations
import Makie

Makie.plottype(::TensorNetwork) = GraphPlot

function Makie.plot!(P::GraphPlot{Tuple{TensorNetwork{A}}}; kwargs...) where {A<:Ansatz}
    tn = P[1][]

    pos = IdDict(tensor => i for (i, tensor) in enumerate(tensors(tn)))
    graph = SimpleGraph([Edge(pos[a], pos[b]) for ind in inds(tn) for (a, b) in combinations(links(ind), 2)])

    scene = Makie.parent_scene(P)
    default_attrs = Makie.default_theme(scene, GraphPlot)

    kwargs = Dict{Symbol,Any}(kwargs)
    if P.attributes.node_size[] == default_attrs.node_size[]
        kwargs[:node_size] = [max(10, log2(size(tensors(tn, i)) |> prod)) for i in 1:nv(graph)]
    end

    graphplot!(P, graph; kwargs...)
end
using Graphs: SimpleGraph, Edge
using GraphMakie: graphplot, GraphPlot
using Combinatorics: combinations
import Makie

Makie.plottype(::TensorNetwork) = GraphPlot

function Makie.plot!(P::GraphPlot{Tuple{GenericTensorNetwork}}; kwargs...)
    tn = P[1][]
    graph = SimpleGraph([Edge(a, b) for ind in inds(tn) for (a, b) in combinations(collect(tn.ind_map[ind]), 2)])

    kwargs = Dict{Symbol,Any}(kwargs)
    get!(kwargs, :node_size) do
        [max(10, log2(size(tensors(tn, i)) |> prod)) for i in 1:nv(graph)]
    end

    graphplot!(P, graph; kwargs...)
end

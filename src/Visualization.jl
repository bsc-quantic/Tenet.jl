using Graphs: SimpleGraph, Edge, edges, ne, nv
using GraphMakie: graphplot!, GraphPlot, to_colormap, get_node_plot
using Combinatorics: combinations
using GraphMakie.NetworkLayout: IterativeLayout
import Makie

function Makie.plot(tn::TensorNetwork{A}; kwargs...) where {A<:Ansatz}
    f = Makie.Figure()

    p, ax = Makie.plot!(f[1,1], tn; kwargs...)
    display(f)

    return f, ax, p
end

function Makie.plot!(f::Makie.GridPosition, tn::TensorNetwork{A}; labels = false, kwargs...) where {A<:Ansatz}
    scene = Makie.Scene()
    default_attrs = Makie.default_theme(scene, GraphPlot)

    kwargs = Dict{Symbol,Any}(kwargs)

    pos = IdDict(tensor => i for (i, tensor) in enumerate(tensors(tn)))
    graph = SimpleGraph([Edge(pos[a], pos[b]) for ind in inds(tn) for (a, b) in combinations(links(ind), 2)])

    kwargs[:node_size] = [max(10, log2(size(tensors(tn,i)) |> prod)) for i in 1:nv(graph)]
    elabels = [join(tn.tensors[edge.src].labels ∩ tn.tensors[edge.dst].labels) for edge in edges(graph)]

    if haskey(kwargs, :layout) && kwargs[:layout] isa IterativeLayout{3}
        ax = Makie.LScene(f[1,1])
    else
        ax = Makie.Axis(f[1,1])
        # hide decorations if it is not a 3D plot
        Makie.hidedecorations!(ax)
        Makie.hidespines!(ax)
        ax.aspect = Makie.DataAspect()
    end

    p = graphplot!(f[1,1], graph;
        elabels = labels ? elabels : nothing,
        elabels_color = [:black for i in 1:ne(graph)],
        # TODO configurable `elabels_textsize`
        elabels_textsize = [17 for i in 1:ne(graph)],
        kwargs...)

    return p, ax
end
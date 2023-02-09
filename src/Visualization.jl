using Graphs: SimpleGraph, Edge, nv
using GraphMakie: graphplot!, GraphPlot, to_colormap, get_node_plot
using Combinatorics: combinations
using GraphMakie.NetworkLayout: IterativeLayout
import Makie

const MAX_NODE_SIZE = 20.

function Makie.plot(tn::TensorNetwork{A}; kwargs...) where {A<:Ansatz}
    f = Makie.Figure()

    p, ax = Makie.plot!(f[1,1], tn; kwargs...)
    display(f)

    return f, ax, p
end

function Makie.plot!(f::Makie.GridPosition, tn::TensorNetwork{A}; kwargs...) where {A<:Ansatz}
    scene = Makie.Scene()
    default_attrs = Makie.default_theme(scene, GraphPlot)

    kwargs = Dict{Symbol,Any}(kwargs)

    pos = IdDict(tensor => i for (i, tensor) in enumerate(tensors(tn)))
    graph = SimpleGraph([Edge(pos[a], pos[b]) for ind in inds(tn) for (a, b) in combinations(links(ind), 2)])

    log_size = [log2(size(tensors(tn, i)) |> prod) for i in 1:nv(graph)]

    min_size, max_size = extrema(log_size)

    kwargs[:node_size] = (log_size/max_size) * MAX_NODE_SIZE

    if haskey(kwargs, :layout) && kwargs[:layout] isa IterativeLayout{3}
        ax = Makie.LScene(f[1,1])
    else
        ax = Makie.Axis(f[1,1])
        # hide decorations if it is not a 3D plot
        Makie.hidedecorations!(ax)
        Makie.hidespines!(ax)
        ax.aspect = DataAspect()
    end

    p = graphplot!(f[1,1], graph; kwargs...)

    return p, ax
end
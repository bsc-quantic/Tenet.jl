module MakieExt

if isdefined(Base, :get_extension)
    using Tenet
else
    using ..Tenet
end

using Graphs: SimpleGraph, Edge, edges, ne, nv, add_edge!, add_vertex!, src, dst
using GraphMakie: graphplot, graphplot!, to_colormap, get_node_plot
using Combinatorics: combinations
using GraphMakie.NetworkLayout: IterativeLayout
import Makie: Axis, AxisPlot, FigureAxisPlot

function __plot_edge_labels(tn::TensorNetwork{A}, graph, copytensors, ghostnodes, opencounter) where {A}
    elabels = Vector{String}([])

    for edge in edges(graph)
        copies = filter(x -> x ∈ copytensors, [edge.src, edge.dst])
        notghosts = filter(x -> x ∉ ghostnodes, [edge.src, edge.dst])

        # TODO refactor this code
        if length(notghosts) == 2 # there are no ghost nodes in this edge
            if isempty(copies) # there are no copy tensors in the nodes of this edge
                push!(elabels, join(Tenet.labels(tensors(tn)[src(edge)]) ∩ Tenet.labels(tensors(tn)[dst(edge)]), ','))
            else
                push!(elabels, string(tensors(tn)[copies[]].meta[:dual]))
            end
        else
            tensor_oinds = filter(id -> nameof(id) ∈ Tenet.labels(tensors(tn)[only(notghosts)]), openinds(tn))
            indices = nameof.(tensor_oinds)

            push!(elabels, string(indices[opencounter[only(notghosts)]]))
            opencounter[only(notghosts)] += 1
        end
    end

    return elabels
end

function __plot_group_tensors(tn::TensorNetwork{A}) where {A}
    pos = IdDict(tensor => i for (i, tensor) in enumerate(tensors(tn)))
    graph = SimpleGraph([Edge(pos[a], pos[b]) for ind in inds(tn) for (a, b) in combinations(links(ind), 2)])

    # TODO recognise them by using `DeltaArray` or `Diagonal` representations
    copytensors = findall(t -> haskey(t.meta, :dual), tensors(tn))

    opentensors = findall(t -> !isempty(Tenet.labels(t) ∩ nameof.(openinds(tn))), tensors(tn))

    opencounter = IdDict(tensor => 1 for tensor in opentensors)
    ghostnodes = map(openinds(tn)) do ind
        add_vertex!(graph)
        node = nv(graph) # TODO is this the best way to get the id of the newly created node?
        tensor = only(links(ind))
        add_edge!(graph, node, pos[tensor])
        return node
    end

    return graph, copytensors, ghostnodes, opencounter
end

function __plot_graph_kwargs(tn::TensorNetwork{A}, graph, copytensors, ghostnodes, kwargs) where {A}
    kwargs[:node_size] = [i ∈ ghostnodes ? 0 : max(15, log2(size(tensors(tn, i)) |> prod)) for i in 1:nv(graph)]
    kwargs[:node_marker] = [i ∈ copytensors ? :diamond : :circle for i in 1:nv(graph)]
    kwargs[:node_color] = [i ∈ copytensors ? :black : :white for i in 1:nv(graph)]

    return kwargs
end

function Makie.plot(tn::TensorNetwork{A}; labels = false, kwargs...) where {A}
    tn = transform(tn, HyperindConverter)
    graph, copytensors, ghostnodes, opencounter = __plot_group_tensors(tn)

    kwargs = __plot_graph_kwargs(tn, graph, copytensors, ghostnodes, Dict{Symbol,Any}(kwargs))

    f, ax, p = graphplot(
        graph;
        elabels = labels ? __plot_edge_labels(tn, graph, copytensors, ghostnodes, opencounter) : nothing,
        # TODO configurable `elabels_textsize`
        elabels_textsize = [17 for i in 1:ne(graph)],
        node_attr = (colormap = :viridis, strokewidth = 2.0, strokecolor = :black),
        kwargs...,
    )

    if ax isa Axis # hide decorations if it is not a 3D plot
        Makie.hidedecorations!(ax)
        Makie.hidespines!(ax)
        ax.aspect = Makie.DataAspect()
    end

    return FigureAxisPlot(f, ax, p)
end

function Makie.plot!(f::Makie.GridPosition, tn::TensorNetwork{A}; labels = false, kwargs...) where {A}
    tn = transform(tn, HyperindConverter)
    graph, copytensors, ghostnodes, opencounter = __plot_group_tensors(tn)

    kwargs = __plot_graph_kwargs(tn, graph, copytensors, ghostnodes, Dict{Symbol,Any}(kwargs))

    if haskey(kwargs, :layout) && kwargs[:layout] isa IterativeLayout{3}
        ax = Makie.LScene(f[1, 1])
    else
        ax = Makie.Axis(f[1, 1])
        # hide decorations if it is not a 3D plot
        Makie.hidedecorations!(ax)
        Makie.hidespines!(ax)
        ax.aspect = Makie.DataAspect()
    end

    p = graphplot!(
        f[1, 1],
        graph;
        elabels = labels ? __plot_edge_labels(tn, graph, copytensors, ghostnodes, opencounter) : nothing,
        # TODO configurable `elabels_textsize`
        elabels_textsize = [17 for i in 1:ne(graph)],
        node_attr = (colormap = :viridis, strokewidth = 2.0, strokecolor = :black),
        kwargs...,
    )

    return AxisPlot(ax, p)
end

end
using Graphs: SimpleGraph, Edge, edges, ne, nv
using GraphMakie: graphplot!, GraphPlot, to_colormap, get_node_plot
using Combinatorics: combinations
using GraphMakie.NetworkLayout: IterativeLayout
import Makie

function Makie.plot(tn::TensorNetwork{A}; kwargs...) where {A<:Ansatz}
    f = Makie.Figure()

    p, ax = Makie.plot!(f[1, 1], tn; kwargs...)
    # display(f)

    # return f, ax, p
    return f
end

function Makie.plot!(f::Makie.GridPosition, tn::TensorNetwork{A}; labels = false, kwargs...) where {A<:Ansatz}
    # scene = Makie.Scene()
    # default_attrs = Makie.default_theme(scene, GraphPlot)

    tn = transform(tn, HyperindConverter)

    kwargs = Dict{Symbol,Any}(kwargs)

    oinds = openinds(tn)
    while !isempty(oinds)
        for ind in oinds
            tensor = links(ind)[1]
            tensor_oinds = filter(id -> id.name ∈ tensor.labels, oinds)
            data = rand(Tuple(id.size for id in tensor_oinds)...)
            indices = (id -> id.name).(tensor_oinds)
            tensor = Tensor(data, indices; out = true)
            push!(tn, tensor)

            filter!(id -> id ∉ tensor_oinds, oinds)
        end
    end

    pos = IdDict(tensor => i for (i, tensor) in enumerate(tensors(tn)))
    graph = SimpleGraph([Edge(pos[a], pos[b]) for ind in inds(tn) for (a, b) in combinations(links(ind), 2)])

    # TODO recognise them by using `DeltaArray` or `Diagonal` representations
    copytensors = findall(t -> haskey(t.meta, :dual), tensors(tn))
    outtensors = findall(t -> haskey(t.meta, :out), tensors(tn))

    kwargs[:node_size] = [i ∈ outtensors ? 0 : max(15, log2(size(tensors(tn, i)) |> prod)) for i in 1:nv(graph)]
    kwargs[:node_marker] = [i ∈ copytensors ? :diamond : :circle for i in 1:length(tensors(tn))]
    kwargs[:node_color] = [i ∈ copytensors ? :black : :white for i in 1:length(tensors(tn))]

    if labels
        elabels = Vector{String}([])
        elabels_color = Vector{Symbol}([])

        for edge in edges(graph)
            copies = filter((x -> x ∈ copytensors), [edge.src, edge.dst])

            if isempty(copies) # there are no copy tensors in the nodes of this edge
                push!(elabels, join(Tenet.labels(tensors(tn)[edge.src]) ∩ Tenet.labels(tensors(tn)[edge.dst]), ','))
                push!(elabels_color, :black)
            else
                push!(elabels, string(tensors(tn)[copies[]].meta[:dual]))
                push!(elabels_color, :grey)
            end
        end
    end

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
        elabels = labels ? elabels : nothing,
        # TODO configurable `elabels_textsize`
        elabels_textsize = [17 for i in 1:ne(graph)],
        node_attr = (colormap = :viridis, strokewidth = 2.0, strokecolor = :black),
        kwargs...,
    )

    return p, ax
end
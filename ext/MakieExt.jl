module MakieExt

if isdefined(Base, :get_extension)
    using Tenet
else
    using ..Tenet
end

using Combinatorics: combinations
using Graphs
using Makie
using NetworkLayout: dim
using GraphMakie

function Makie.plot(tn::TensorNetwork; kwargs...)
    f = Figure()
    ax, p = plot!(f[1, 1], tn; kwargs...)
    return Makie.FigureAxisPlot(f, ax, p)
end

function Makie.plot!(f::Union{Figure,GridPosition}, tn::TensorNetwork; kwargs...)
    ax = if haskey(kwargs, :layout) && dim(kwargs[:layout]) == 3
        Axis3(f[1, 1])
    else
        ax = Axis(f[1, 1])
        ax.aspect = DataAspect()
        ax
    end

    hidedecorations!(ax)
    hidespines!(ax)

    p = plot!(ax, tn; kwargs...)

    return Makie.AxisPlot(ax, p)
end

function Makie.plot!(ax::Union{Axis,Axis3}, tn::TensorNetwork; labels = false, kwargs...)
    tn = transform(tn, Tenet.HyperindConverter)

    # TODO how to mark multiedges? (i.e. parallel edges)
    handles = IdDict(obj => i for (i, obj) in enumerate(tensors(tn)))
    graph = SimpleGraph([Edge(handles[a], handles[b]) for ind in inds(tn) for (a, b) in combinations(links(ind), 2)])

    # TODO recognise `copytensors` by using `DeltaArray` or `Diagonal` representations
    copytensors = findall(t -> haskey(t.meta, :dual), tensors(tn))
    ghostnodes = map(openinds(tn)) do ind
        add_vertex!(graph)
        node = nv(graph) # TODO is this the best way to get the id of the newly created node?
        tensor = only(links(ind))
        add_edge!(graph, node, handles[tensor])
        return node
    end

    # configure graphics
    # TODO refactor hardcoded values into constants
    kwargs = Dict{Symbol,Any}(kwargs)

    get!(kwargs, :node_size) do
        map(1:nv(graph)) do i
            if i ∈ ghostnodes
                0
            else
                max(15, log2(length(tensors(tn)[i])))
            end
        end
    end
    get!(() -> map(i -> i ∈ copytensors ? :diamond : :circle, 1:nv(graph)), kwargs, :node_marker)
    get!(() -> map(i -> i ∈ copytensors ? :black : :white, 1:nv(graph)), kwargs, :node_color)

    get!(kwargs, :node_attr, (colormap = :viridis, strokewidth = 2.0, strokecolor = :black))

    # configure labels
    labels == true && get!(kwargs, :elabels) do
        opentensors = findall(t -> !isdisjoint(Tenet.labels(t), Tenet.labels(tn, :open)), tensors(tn))
        opencounter = IdDict(tensor => 0 for tensor in opentensors)

        map(edges(graph)) do edge
            # case: open edge
            if any(∈(ghostnodes), [src(edge), dst(edge)])
                notghost = src(edge) ∈ ghostnodes ? dst(edge) : src(edge)
                inds = nameof.(openinds(tn)) ∩ Tenet.labels(tensors(tn)[notghost])
                opencounter[notghost] += 1
                return inds[opencounter[notghost]] |> string
            end

            # case: hyperedge
            if any(∈(copytensors), [src(edge), dst(edge)])
                i = src(edge) ∈ copytensors ? src(edge) : dst(edge)
                return tensors(tn)[i].meta[:dual] |> string
            end

            return join(Tenet.labels(tensors(tn)[src(edge)]) ∩ Tenet.labels(tensors(tn)[dst(edge)]), ',')
        end
    end
    get!(() -> repeat([:black], ne(graph)), kwargs, :elabels_color)
    get!(() -> repeat([17], ne(graph)), kwargs, :elabels_textsize)

    # plot graph
    graphplot!(ax, graph; kwargs...)
end

end

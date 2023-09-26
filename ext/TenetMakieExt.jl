module TenetMakieExt

using Tenet
using Combinatorics: combinations
using Graphs
using Makie

using GraphMakie

"""
    plot(tn::TensorNetwork; kwargs...)
    plot!(f::Union{Figure,GridPosition}, tn::TensorNetwork; kwargs...)
    plot!(ax::Union{Axis,Axis3}, tn::TensorNetwork; kwargs...)

Plot a [`TensorNetwork`](@ref) as a graph.

# Keyword Arguments

  - `labels` If `true`, show the labels of the tensor indices. Defaults to `false`.
  -  The rest of `kwargs` are passed to `GraphMakie.graphplot`.
"""
function Makie.plot(@nospecialize tn::TensorNetwork; kwargs...)
    f = Figure()
    ax, p = plot!(f[1, 1], tn; kwargs...)
    return Makie.FigureAxisPlot(f, ax, p)
end

# NOTE this is a hack! we did it in order not to depend on NetworkLayout but can be unstable
__networklayout_dim(x) = typeof(x).super.parameters |> first

function Makie.plot!(f::Union{Figure,GridPosition}, @nospecialize tn::TensorNetwork; kwargs...)
    ax = if haskey(kwargs, :layout) && __networklayout_dim(kwargs[:layout]) == 3
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

function Makie.plot!(ax::Union{Axis,Axis3}, @nospecialize tn::TensorNetwork; labels = false, kwargs...)
    hypermap = Tenet.hyperflatten(tn)
    tn = transform(tn, Tenet.HyperindConverter)

    # TODO how to mark multiedges? (i.e. parallel edges)
    graph = SimpleGraph([Edge(tensors...) for (_, tensors) in tn.indices if length(tensors) > 1])

    # TODO recognise `copytensors` by using `DeltaArray` or `Diagonal` representations
    copytensors = findall(tensor -> any(flatinds -> issetequal(inds(tensor), flatinds), keys(hypermap)), tensors(tn))
    ghostnodes = map(inds(tn, :open)) do ind
        # create new ghost node
        add_vertex!(graph)
        node = nv(graph)

        # connect ghost node
        tensor = only(tn.indices[ind])
        add_edge!(graph, node, tensor)

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
    get!(
        () -> map(i -> i ∈ copytensors ? :black : Makie.RGBf(240 // 256, 180 // 256, 100 // 256), 1:nv(graph)),
        kwargs,
        :node_color,
    )

    get!(kwargs, :node_attr, (colormap = :viridis, strokewidth = 2.0, strokecolor = :black))

    # configure labels
    labels == true && get!(kwargs, :elabels) do
        opentensors = findall(t -> !isdisjoint(inds(t), inds(tn, :open)), tensors(tn))
        opencounter = IdDict(tensor => 0 for tensor in opentensors)

        map(edges(graph)) do edge
            # case: open edge
            if any(∈(ghostnodes), [src(edge), dst(edge)])
                notghost = src(edge) ∈ ghostnodes ? dst(edge) : src(edge)
                inds = Tenet.inds(tn, :open) ∩ Tenet.inds(tensors(tn)[notghost])
                opencounter[notghost] += 1
                return inds[opencounter[notghost]] |> string
            end

            # case: hyperedge
            if any(∈(copytensors), [src(edge), dst(edge)])
                i = src(edge) ∈ copytensors ? src(edge) : dst(edge)
                # hyperindex = filter(p -> isdisjoint(inds(tensors)[i], p[2]), hypermap) |> only |> first
                hyperindex = hypermap[Tenet.inds(tensors(tn)[i])]
                return hyperindex |> string
            end

            return join(Tenet.inds(tensors(tn)[src(edge)]) ∩ Tenet.inds(tensors(tn)[dst(edge)]), ',')
        end
    end
    get!(() -> repeat([:black], ne(graph)), kwargs, :elabels_color)
    get!(() -> repeat([17], ne(graph)), kwargs, :elabels_textsize)

    # plot graph
    graphplot!(ax, graph; kwargs...)
end

end

module TenetGraphMakieExt

using Tenet
using GraphMakie
using Graphs: Graphs, vertices
using Makie
using Combinatorics: combinations
const NetworkLayout = GraphMakie.NetworkLayout

"""
    graphplot(tn::TensorNetwork; kwargs...)
    graphplot!(f::Union{Figure,GridPosition}, tn::TensorNetwork; kwargs...)
    graphplot!(ax::Union{Axis,Axis3}, tn::TensorNetwork; kwargs...)

Plot a [`TensorNetwork`](@ref) as a graph.

# Keyword Arguments

  - `labels` If `true`, show the labels of the tensor indices. Defaults to `false`.
  - The rest of `kwargs` are passed to `GraphMakie.graphplot`.
"""
function GraphMakie.graphplot(tn::Union{Tenet.AbstractTensorNetwork,Lattice}; kwargs...)
    f = Figure()
    ax, p = graphplot!(f[1, 1], tn; kwargs...)
    return Makie.FigureAxisPlot(f, ax, p)
end

# NOTE this is a hack! we did it in order not to depend on NetworkLayout but can be unstable
__networklayout_dim(x) = first(typeof(x).super.parameters)

function GraphMakie.graphplot!(f::Union{Figure,GridPosition}, tn::Union{Tenet.AbstractTensorNetwork,Lattice}; kwargs...)
    ax = if haskey(kwargs, :layout) && __networklayout_dim(kwargs[:layout]) == 3
        Axis3(f[1, 1])
    else
        ax = Axis(f[1, 1])
        ax.aspect = DataAspect()
        ax
    end

    hidedecorations!(ax)
    hidespines!(ax)

    p = graphplot!(ax, tn; kwargs...)

    return Makie.AxisPlot(ax, p)
end

function GraphMakie.graphplot!(ax::Union{Axis,Axis3}, tn::Tenet.AbstractTensorNetwork; labels=false, kwargs...)
    tn = TensorNetwork(tn)
    hypermap = Tenet.hyperflatten(tn)
    tn = transform(tn, Tenet.HyperFlatten)

    tensormap = IdDict(tensor => i for (i, tensor) in enumerate(tensors(tn)))

    graph = Graphs.SimpleGraph(length(tensors(tn)))
    for i in setdiff(inds(tn; set=:inner), inds(tn; set=:hyper))
        edge_tensors = tensors(tn; intersects=i)

        @assert length(edge_tensors) == 2
        a, b = edge_tensors

        Graphs.add_edge!(graph, tensormap[a], tensormap[b])
    end

    # TODO recognise `copytensors` by using `DeltaArray` or `Diagonal` representations
    copytensors = findall(tensor -> any(flatinds -> issetequal(inds(tensor), flatinds), keys(hypermap)), tensors(tn))
    ghostnodes = map(inds(tn; set=:open)) do index
        # create new ghost node
        Graphs.add_vertex!(graph)
        node = Graphs.nv(graph)

        # connect ghost node
        tensor = only(tn.indexmap[index])
        Graphs.add_edge!(graph, node, tensormap[tensor])

        return node
    end

    # configure graphics
    # TODO refactor hardcoded values into constants
    kwargs = Dict{Symbol,Any}(kwargs)

    if haskey(kwargs, :node_size)
        append!(kwargs[:node_size], zero(ghostnodes))
    else
        kwargs[:node_size] = map(1:Graphs.nv(graph)) do i
            i ∈ ghostnodes ? 0 : max(15, log2(length(tensors(tn)[i])))
        end
    end

    if haskey(kwargs, :node_marker)
        append!(kwargs[:node_marker], fill(:circle, length(ghostnodes)))
    else
        kwargs[:node_marker] = map(i -> i ∈ copytensors ? :diamond : :circle, 1:Graphs.nv(graph))
    end

    if haskey(kwargs, :node_color)
        kwargs[:node_color] = vcat(kwargs[:node_color], fill(:black, length(ghostnodes)))
    else
        kwargs[:node_color] = map(1:Graphs.nv(graph)) do v
            v ∈ copytensors ? Makie.to_color(:black) : Makie.RGBf(240//256, 180//256, 100//256)
        end
    end

    get!(kwargs, :node_attr, (colormap=:viridis, strokewidth=2.0, strokecolor=:black))

    # configure labels
    labels == true && get!(kwargs, :elabels) do
        opentensors = findall(t -> !isdisjoint(inds(t), inds(tn; set=:open)), tensors(tn))
        opencounter = IdDict(tensor => 0 for tensor in opentensors)

        map(Graphs.edges(graph)) do edge
            # case: open edge
            if any(∈(ghostnodes), [Graphs.src(edge), Graphs.dst(edge)])
                notghost = Graphs.src(edge) ∈ ghostnodes ? Graphs.dst(edge) : Graphs.src(edge)
                inds = Tenet.inds(tn; set=:open) ∩ Tenet.inds(tensors(tn)[notghost])
                opencounter[notghost] += 1
                return string(inds[opencounter[notghost]])
            end

            # case: hyperedge
            if any(∈(copytensors), [Graphs.src(edge), Graphs.dst(edge)])
                i = Graphs.src(edge) ∈ copytensors ? Graphs.src(edge) : Graphs.dst(edge)
                # hyperindex = filter(p -> isdisjoint(inds(tensors)[i], p[2]), hypermap) |> only |> first
                hyperindex = hypermap[Tenet.inds(tensors(tn)[i])]
                return string(hyperindex)
            end

            return join(Tenet.inds(tensors(tn)[Graphs.src(edge)]) ∩ Tenet.inds(tensors(tn)[Graphs.dst(edge)]), ',')
        end
    end
    get!(() -> repeat([:black], Graphs.ne(graph)), kwargs, :elabels_color)
    get!(() -> repeat([17], Graphs.ne(graph)), kwargs, :elabels_textsize)
    get!(() -> NetworkLayout.Stress(), kwargs, :layout)

    # plot graph
    return graphplot!(ax, graph; kwargs...)
end

function GraphMakie.graphplot!(ax::Makie.AbstractAxis, lattice::Lattice; labels=false, kwargs...)
    kwargs = Dict{Symbol,Any}(kwargs)
    labels == true && get!(kwargs, :ilabels) do
        return collect(string.(vertices(lattice)))
    end
    return graphplot!(ax, parent(lattice); kwargs...)
end

end

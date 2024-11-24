using Graphs: Graphs, vertices, edges
using OrderedCollections: OrderedDict

function graph_representation(tn::AbstractTensorNetwork)
    tn = TensorNetwork(tn)
    hypermap = Tenet.hyperflatten(tn)
    if !isempty(hypermap)
        tn = transform(tn, Tenet.HyperFlatten)
    end
    tensormap = IdDict(tensor => i for (i, tensor) in enumerate(tensors(tn)))

    graph = Graphs.SimpleGraph(length(tensors(tn)))
    for i in setdiff(inds(tn; set=:inner), inds(tn; set=:hyper))
        a, b = tensors(tn; intersects=i)
        Graphs.add_edge!(graph, tensormap[a], tensormap[b])
    end

    # TODO recognise `copytensors` by using `DeltaArray` or `Diagonal` representations
    hypernodes = findall(tensor -> any(flatinds -> issetequal(inds(tensor), flatinds), keys(hypermap)), tensors(tn))
    ghostnodes = map(inds(tn; set=:open)) do index
        # create new ghost node
        Graphs.add_vertex!(graph)
        node = Graphs.nv(graph)

        # connect ghost node
        tensor = only(tn.indexmap[index])
        Graphs.add_edge!(graph, node, tensormap[tensor])

        return node
    end

    return tn, graph, tensormap, hypermap, hypernodes, ghostnodes
end

# TODO use `Base.ENV["VSCODE_PID"]` to detect if running in vscode notebook?
# TODO move this to VSCodeServer.jl
function Base.Multimedia.istextmime(::MIME"application/vnd.vega.v5+json")
    return true
end

Base.show(io::IO, ::MIME"application/vnd.vega.v5+json", @nospecialize(tn::AbstractTensorNetwork)) = draw(io, tn)

@inline draw(io::IO, args...) = draw(IOContext(io), args...)

function draw(io::IOContext, @nospecialize(tn::AbstractTensorNetwork))
    width = get(io, :width, 800)
    height = get(io, :height, 600)

    tn, graph, tensormap, hypermap, hypernodes, ghostnodes = graph_representation(tn)
    hypermap = Dict(Iterators.flatten([[i => v for i in k] for (k, v) in hypermap]))

    spec = OrderedDict{Symbol,Any}()
    spec[Symbol("\$schema")] = "https://vega.github.io/schema/vega/v5.json"
    spec[:width] = width
    spec[:height] = height
    spec[:signals] = [
        OrderedDict(:name => "center.x", :update => "width / 2", :__collapse => true),
        OrderedDict(:name => "center.y", :update => "height / 2", :__collapse => true),
    ]
    vdata = OrderedDict{Symbol,Any}()
    edata = OrderedDict{Symbol,Any}()
    spec[:data] = [vdata, edata]
    vmarks = OrderedDict{Symbol,Any}()
    emarks = OrderedDict{Symbol,Any}()
    spec[:marks] = [vmarks, emarks]

    # vertex data
    vdata[:name] = "vertex-data"
    vdata[:format] = OrderedDict(:type => "json", :__collapse => true)
    vdata[:values] = map(vertices(graph)) do v
        group = if v ∈ ghostnodes
            0
        elseif v ∈ hypernodes
            1
        else
            2
        end
        OrderedDict(:name => v, :group => group, :__collapse => true)
    end

    # edge data
    edata[:name] = "edge-data"
    edata[:format] = OrderedDict(:type => "json", :__collapse => true)
    edata[:values] = map(inds(tn)) do i
        nodes = tensors(tn; intersects=i)

        if length(nodes) == 2
            # regular nodes
            a, b = nodes
            index = get(hypermap, i, i)
            OrderedDict(
                :source => tensormap[a] - 1,
                :target => tensormap[b] - 1,
                :value => string(index),
                :__collapse => true,
            )

        elseif length(nodes) == 1
            # nodes with open indices
            vertex = tensormap[only(nodes)] - 1
            ghost = ntensors(tn) + vertex
            OrderedDict(:source => vertex, :target => ghost, :value => string(i), :__collapse => true)
        end
    end

    # vertex marks
    vmarks[:name] = "vertices"
    vmarks[:type] = "symbol"
    vmarks[:from] = OrderedDict(:data => "vertex-data", :__collapse => true)
    vmarks[:transform] = [
        OrderedDict(
            :type => "force",
            :static => false,
            :iterations => 1000,
            :forces => [
                OrderedDict(
                    :force => "center",
                    :x => OrderedDict(:signal => "center.x", :__collapse => true),
                    :y => OrderedDict(:signal => "center.y", :__collapse => true),
                    :__collapse => true,
                ),
                OrderedDict(:force => "collide", :radius => 10, :__collapse => true),
                OrderedDict(:force => "nbody", :strength => -100, :__collapse => true),
                OrderedDict(:force => "link", :links => "edge-data", :distance => 100, :__collapse => true),
            ],
        ),
    ]

    # edge marks
    emarks[:type] = "path"
    emarks[:from] = OrderedDict(:data => "edge-data", :__collapse => true)
    emarks[:transform] = [
        OrderedDict(
            :type => "linkpath",
            :shape => "line",
            :sourceX => "datum.source.x",
            :sourceY => "datum.source.y",
            :targetX => "datum.target.x",
            :targetY => "datum.target.y",
        ),
    ]

    return print_json(io, spec)
end

function print_json(io::IO, spec; indent::Int=0)
    spec = copy(spec)
    collapse = get(spec, :__collapse, false)
    delete!(spec, :__collapse)

    print(io, '\t'^indent * "{" * (collapse ? "" : "\n"))

    for (i, (key, value)) in enumerate(spec)
        print(io, (collapse ? "" : '\t'^(indent + 1)) * "\"$key\": ")

        if isa(value, AbstractDict)
            print_json(io, value; indent=0)

        elseif isa(value, Vector)
            print(io, "[")
            for (j, v) in enumerate(value)
                print(io, collapse ? "" : "\n")
                print_json(io, v; indent=indent + 2)

                islast = j == length(value)
                print(io, (islast ? "" : ","))
            end
            print(io, "\n" * '\t'^(indent + 1) * "]")

        elseif isa(value, AbstractString)
            print(io, "\"$value\"")

        else
            print(io, value)
        end

        islast = i == length(spec)
        print(io, (islast ? "" : ",") * (collapse ? "" : "\n"))
    end

    print(io, (collapse ? "" : '\t'^indent) * "}")
end

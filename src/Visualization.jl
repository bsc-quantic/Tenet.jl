using Graphs: Graphs, vertices, edges

function graph_representation(tn::AbstractTensorNetwork)
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

# TODO use `Base.ENV["VSCODE_PID"]` to detect if running in vscode notebook
# Base.show(io::IO, ::MIME"text/html", @nospecialize(tn::AbstractTensorNetwork)) = draw(io, tn)
# Base.show(io::IO, ::MIME"juliavscode/html", @nospecialize(tn::AbstractTensorNetwork)) = draw(io, tn)

for v in ["application/vnd.vegalite.v3+json", "application/vnd.vegalite.v4+json", "application/vnd.vegalite.v5+json"]
    @eval Base.Multimedia.istextmime(::MIME{$(Symbol(v))}) = true
end

Base.show(io::IO, ::MIME"application/vnd.vegalite.v5+json", @nospecialize(tn::AbstractTensorNetwork)) = draw(io, tn)
function draw(io::IO, @nospecialize(tn::AbstractTensorNetwork))
    tn, graph, tensormap, hypermap, hypernodes, ghostnodes = graph_representation(tn)
    hypermap = Dict(Iterators.flatten([[i => v for i in k] for (k, v) in hypermap]))

    json = """{
            "\$schema": "https://vega.github.io/schema/vega/v5.json",
            "width": 600,
            "height": 600,
            "signals": [
                {"name": "center.x", "update": "width / 2"},
                {"name": "center.y", "update": "height / 2"},
            ],
            "data": [
                {
                    "name": "vertex-data",
                    "format": {"type": "json" },
                    "values": [\n\t\t$(
                        join(map(vertices(graph)) do v
                            return "{ \"name\": \"$v\", \"group\": $(
                            if v ∈ ghostnodes
                                0 # "\"ghost\""
                            elseif v ∈ hypernodes
                                1 # "\"hyper\""
                            else
                                2 # "\"tensor\""
                            end) }"
                        end,
                        ",\n\t\t"))
                    ]
                },
                {
                    "name": "edge-data",
                    "format": {"type": "json" },
                    "values": [\n\t\t$(
                        join(map(inds(tn)) do i
                            # NOTE Vega uses 0-indexing
                            nodes = tensors(tn; intersects=i)

                            # regular nodes
                            if length(nodes) == 2
                                a, b = nodes
                                index = get(hypermap, i, i)
                                return "{ \"source\": $(tensormap[a] - 1), \"target\": $(tensormap[b] - 1), \"value\": \"$index\" }"

                            # nodes with open indices
                            elseif length(nodes) == 1
                                vertex = tensormap[only(nodes)] - 1
                                # TODO is this correct? what if more than 1 open index (e.g. MPO)
                                ghost = ntensors(tn) + vertex
                                return "{ \"source\": $vertex, \"target\": $ghost, \"value\": \"$i\" }"
                            end
                        end,
                        ",\n\t\t")
                        )
                    ]
                }
            ],
            "marks": [
                {
                    "name": "vertices",
                    "type": "symbol",
                    "from": {"data": "vertex-data"},
                    "encode": {
                    },
                    "transform": [
                        {
                            "type": "force",
                            "static": false,
                            "iterations": 1000,
                            "forces": [
                                {"force": "center", "x": {"signal": "center.x"}, "y": {"signal": "center.y"}},
                                {"force": "collide", "radius": 10},
                                {"force": "nbody", "strength": -100},
                                {"force": "link", "links": "edge-data", "distance": 100},
                            ]
                        }
                    ]
                },
                {
                    "type": "path",
                    "from": {"data": "edge-data"},
                    "encode": {
                        "update": {
                            "stroke": {"value": "#ccc"},
                            "strokeWidth": {"value": 2},
                            "tooltip": {"field": "name"}
                        }
                    },
                    "transform": [
                        {
                            "type": "linkpath",
                            "shape": "line",
                            "sourceX": "datum.source.x",
                            "sourceY": "datum.source.y",
                            "targetX": "datum.target.x",
                            "targetY": "datum.target.y"
                        }
                    ]
                }
            ]
        }"""

    return print(io, json)
end

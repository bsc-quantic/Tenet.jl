using ECharts_jll
using Graphs

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

Base.show(io::IO, ::MIME"text/html", @nospecialize(tn::AbstractTensorNetwork)) = show(io, MIME"juliavscode/html"(), tn)
function Base.show(io::IO, ::MIME"juliavscode/html", @nospecialize(tn::AbstractTensorNetwork))
    tn, graph, tensormap, hypermap, hypernodes, ghostnodes = graph_representation(tn)
    hypermap = Dict(Iterators.flatten([[i => v for i in k] for (k, v) in hypermap]))

    appid = gensym("tenet-graph")
    return print(
        io,
        """
            <script type="text/javascript">$(
                read(joinpath(dirname(ECharts_jll.echarts), "echarts.min.js"), String)
            )</script>
            <div id="$appid" style="width: 600px; height: 600px;"></div>
            <script>
                var chart = echarts.init(document.getElementById('$appid'), null, {renderer: 'svg'});

                option = {
                    series: [
                        {
                            type: 'graph',
                            layout: 'force',
                            animation: false,
                            darkMode: 'auto',
                            left: '5%',
                            top: '5%',
                            width: '95%',
                            height: '95%',
                            draggable: true,
                            force: {
                                gravity: 0,
                                repulsion: 100,
                                edgeLength: 4
                            },
                            data: [$(join(map(vertices(graph)) do v
                                return "{ " *
                                    "name: '$v', " *
                                    "symbol: $(if v ∈ ghostnodes
                                        "'none'"
                                        elseif v ∈ hypernodes
                                            "'diamond'"
                                        else
                                            "'circle'"
                                        end), " *
                                    "symbolSize: $(v ∈ ghostnodes ? "0" : "8")" *
                                " }"
                            end, ", "))],
                            edges: [$(join(
                                map(inds(tn)) do i
                                    nodes = tensors(tn; intersects=i)
                                    if length(nodes) == 1
                                        # TODO
                                        v = tensormap[only(nodes)]
                                        return "{ source: '$v', target: '$v', name: '$i' }"
                                    elseif length(nodes) == 2
                                        a, b = nodes
                                        index = get(hypermap, i, i)
                                        return "{ source: '$(tensormap[a])', target: '$(tensormap[b])', name: '$index' }"
                                    end
                                end
                            , ','))],
                            edgeLabel: {
                                show: false,
                                formatter: function (params) {
                                    return params.data.name;
                                }
                            },
                            emphasis: {
                                label: false,
                                edgeLabel: {
                                    show: true,
                                    fontSize: 20,
                                },
                                lineStyle: {
                                    width: 3
                                },
                            },
                        }
                    ]
                };

                option && chart.setOption(option);
            </script>
        """,
    )
end

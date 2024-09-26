using ECharts_jll
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

Base.show(io::IO, ::MIME"application/vnd.vegalite.v3+json", @nospecialize(tn::AbstractTensorNetwork)) = draw(io, tn)
function draw(io::IO, @nospecialize(tn::AbstractTensorNetwork))
    tn, graph, tensormap, hypermap, hypernodes, ghostnodes = graph_representation(tn)
    hypermap = Dict(Iterators.flatten([[i => v for i in k] for (k, v) in hypermap]))
    return print(
        io,
        """{
            "\$schema": "https://vega.github.io/schema/vega-lite/v3.json",
            "description": "A simple bar chart with embedded data.",
            "data": {
                "values": [
                {"a": "A", "b": 28}, {"a": "B", "b": 55}, {"a": "C", "b": 43},
                {"a": "D", "b": 91}, {"a": "E", "b": 81}, {"a": "F", "b": 53},
                {"a": "G", "b": 19}, {"a": "H", "b": 87}, {"a": "I", "b": 52}
                ]
            },
            "mark": "bar",
            "encoding": {
                "x": {"field": "a", "type": "nominal", "axis": {"labelAngle": 0}},
                "y": {"field": "b", "type": "quantitative"}
            }
        }""",
        # """{
        #     "\$schema": "https://vega.github.io/schema/vega/v5.json",
        #     "width": 600,
        #     "height": 600,
        # "signals": [
        #     {"name": "center.x", "update": "width / 2"},
        #     {"name": "center.y", "update": "height / 2"},
        # ],
        #     "data": [
        #         {
        #             "name": "vertex-data",
        #             "format": {"type": "json" },
        #             "values": [$(
        #                 join(map(vertices(graph)) do v
        #                     return "{ \"name\": \"$v\", \"group\": $(
        #                     if v ∈ ghostnodes
        #                         0 # "\"ghost\""
        #                     elseif v ∈ hypernodes
        #                         1 # "\"hyper\""
        #                     else
        #                         2 # "\"tensor\""
        #                     end) }"
        #                 end,
        #                 ","))],
        #         },
        #         {
        #             "name": "edge-data",
        #             "format": {"type": "json" },
        #             "values": [$(
        #                 join(map(inds(tn)) do i
        #                     nodes = tensors(tn; intersects=i)
        #                     if length(nodes) == 1
        #                         # TODO
        #                         v = tensormap[only(nodes)]
        #                         return "{ source: \"$v\", target: \"$v\", name: \"$i\" }"
        #                     elseif length(nodes) == 2
        #                         a, b = nodes
        #                         index = get(hypermap, i, i)
        #                         return "{ source: \"$(tensormap[a])\", target: \"$(tensormap[b])\", name: \"$index\" }"
        #                     end
        #                 end,
        #                 ','))],
        #         }
        #     ],
        #     "marks": [
        #         {
        #             "name": "vertices",
        #             "type": "symbol",
        #             "from": {"data": "vertex-data"},
        #             "encode": {
        #             },
        #             "transform": [
        #                 {
        #                     "type": "force",
        #                     "static": false,
        #                     "iterations": 100,
        #                     "forces": [
        #                         {"force": "center", "x": {"signal": "center.x"}, "y": {"signal": "center.y"}},
        #                         {"force": "collide", "radius": 10},
        #                         {"force": "nbody", "strength": -100},
        #                         {"force": "link", "links": "edge-data", "distance": 100},
        #                     ]
        #                 }
        #             ]
        #         },
        #         {
        #             "type": "path",
        #             "from": {"data": "edge-data"},
        #             "encode": {
        #                 "update": {
        #                     "stroke": {"value": "#ccc"},
        #                     "strokeWidth": {"value": 2},
        #                     "tooltip": {"field": "name"}
        #                 }
        #             },
        #             "transform": [
        #                 {
        #                     "type": "linkpath",
        #                     "require": {"signal": "force"},
        #                     "shape": "line",
        #                     "sourceX": "datum.source.x",
        #                     "sourceY": "datum.source.y",
        #                     "targetX": "datum.target.x",
        #                     "targetY": "datum.target.y"
        #                 }
        #             ]
        #         }
        #     ]
        # }""",
    )
end

# function draw(io::IO, @nospecialize(tn::AbstractTensorNetwork))
#     tn, graph, tensormap, hypermap, hypernodes, ghostnodes = graph_representation(tn)
#     hypermap = Dict(Iterators.flatten([[i => v for i in k] for (k, v) in hypermap]))

#     appid = gensym("tenet-graph")
#     return print(
#         io,
#         """
#             <script type="text/javascript">$(
#                 read(joinpath(dirname(ECharts_jll.echarts), "echarts.min.js"), String)
#             )</script>
#             <div id="$appid" style="width: 600px; height: 600px;"></div>
#             <script>
#                 var chart = echarts.init(document.getElementById('$appid'), null, {renderer: 'svg'});

#                 option = {
#                     series: [
#                         {
#                             type: 'graph',
#                             layout: 'force',
#                             animation: false,
#                             darkMode: 'auto',
#                             left: '5%',
#                             top: '5%',
#                             width: '95%',
#                             height: '95%',
#                             draggable: true,
#                             force: {
#                                 gravity: 0,
#                                 repulsion: 100,
#                                 edgeLength: 4
#                             },
#                             data: [$(join(map(vertices(graph)) do v
#                                 return "{ " *
#                                     "name: '$v', " *
#                                     "symbol: $(if v ∈ ghostnodes
#                                         "'none'"
#                                         elseif v ∈ hypernodes
#                                             "'diamond'"
#                                         else
#                                             "'circle'"
#                                         end), " *
#                                     "symbolSize: $(v ∈ ghostnodes ? "0" : "8")" *
#                                 " }"
#                             end, ", "))],
#                             edges: [$(join(
#                                 map(inds(tn)) do i
#                                     nodes = tensors(tn; intersects=i)
#                                     if length(nodes) == 1
#                                         # TODO
#                                         v = tensormap[only(nodes)]
#                                         return "{ source: '$v', target: '$v', name: '$i' }"
#                                     elseif length(nodes) == 2
#                                         a, b = nodes
#                                         index = get(hypermap, i, i)
#                                         return "{ source: '$(tensormap[a])', target: '$(tensormap[b])', name: '$index' }"
#                                     end
#                                 end
#                             , ','))],
#                             edgeLabel: {
#                                 show: false,
#                                 formatter: function (params) {
#                                     return params.data.name;
#                                 }
#                             },
#                             emphasis: {
#                                 label: false,
#                                 edgeLabel: {
#                                     show: true,
#                                     fontSize: 20,
#                                 },
#                                 lineStyle: {
#                                     width: 3
#                                 },
#                             },
#                         }
#                     ]
#                 };

#                 option && chart.setOption(option);
#             </script>
#         """,
#     )
# end

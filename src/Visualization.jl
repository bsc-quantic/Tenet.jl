using ECharts_jll

function Base.show(io::IO, ::MIME"juliavscode/html", @nospecialize(tn::AbstractTensorNetwork))
    tn = transform(tn, Tenet.HyperFlatten)
    appid = gensym("tenet-graph")
    tensormap = IdDict([tensor => i for (i, tensor) in enumerate(tensors(tn))])

    nodes = collect(values(tensormap))

    return print(
        io,
        """
            <script type="text/javascript">$(
                read(joinpath(dirname(ECharts_jll.echarts), "echarts.min.js"), String)
            )</script>
            <div id="$appid" style="width: 600px; height: 600px;"></div>
            <script>
                var chart = echarts.init(document.getElementById('$appid'));

                option = {
                    series: [
                        {
                            type: 'graph',
                            layout: 'force',
                            animation: false,
                            left: '0%',
                            top: '0%',
                            width: '100%',
                            height: '100%',
                            draggable: true,
                            force: {
                                // initLayout: 'circular'
                                gravity: 0.1,
                                repulsion: 100,
                                edgeLength: 3
                            },
                            data: $nodes,
                            edges: [$(join(
                                map(inds(tn; set=:inner)) do i
                                    a,b = tensors(tn; intersects=i)
                                    return "{ source: $(tensormap[a]), target: $(tensormap[b]), name: '$i' }"
                                end
                            , ','))],
                            edgeLabel: {
                                show: false,
                                formatter: function (params) {
                                    return params.data.name;
                                }
                            },
                            emphasis: {
                                edgeLabel: {
                                    show: true
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

using Graphs
using GraphMakie
using Combinatorics

function draw(tn::TensorNetwork; kwargs...)
    graph = SimpleGraph([Edge(a, b) for ind in inds(tn) for (a, b) in combinations(collect(tn.ind_map[ind]), 2)])

    kwargs = Dict{Symbol,Any}(kwargs)
    get!(kwargs, :node_size) do
        [max(10, log2(size(tensors(tn, i)) |> prod)) for i in 1:nv(graph)]
    end

    graphplot(graph; kwargs...)
end
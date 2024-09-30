using Combinatorics

struct Dense <: AbstractAnsatz
    tn::Ansatz
end

Ansatz(tn::Dense) = tn.tn

Base.copy(qtn::Dense) = Dense(copy(Ansatz(qtn)))
Base.similar(qtn::Dense) = Dense(similar(Ansatz(qtn)))
Base.zero(qtn::Dense) = Dense(zero(Ansatz(qtn)))

function Dense(::State, array::AbstractArray; sites=Site.(1:ndims(array)))
    n = ndims(array)
    @assert n > 0
    @assert all(>(1), size(array))

    gen = IndexCounter()
    symbols = [nextindex!(gen) for _ in 1:n]
    sitemap = Dict{Site,Symbol}(
        map(sites, 1:n) do site, i
            site => symbols[i]
        end,
    )

    tensor = Tensor(array, symbols)

    tn = TensorNetwork([tensor])
    qtn = Quantum(tn, sitemap)
    graph = complete_graph(nlanes(qtn))
    lattice = MetaGraph(graph, lanes(qtn) .=> nothing, map(x -> Site.(Tuple(x)) => nothing, edges(graph)))
    ansatz = Ansatz(qtn, lattice)
    return Dense(ansatz)
end

function Dense(::Operator, array::AbstractArray; sites)
    n = ndims(array)
    @assert n > 0
    @assert all(>(1), size(array))
    @assert length(sites) == n

    gen = IndexCounter()
    tensor_inds = [nextindex!(gen) for _ in 1:n]
    tensor = Tensor(array, tensor_inds)
    tn = TensorNetwork([tensor])

    sitemap = Dict{Site,Symbol}(map(splat(Pair), zip(sites, tensor_inds)))
    qtn = Quantum(tn, sitemap)
    graph = complete_graph(nlanes(qtn))
    lattice = MetaGraph(graph, lanes(qtn) .=> nothing, map(x -> Site.(Tuple(x)) => nothing, edges(graph)))
    ansatz = Ansatz(qtn, lattice)
    return Dense(ansatz)
end

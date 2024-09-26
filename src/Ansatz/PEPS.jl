abstract type AbstractPEPS <: AbstractAnsatz end

struct PEPS <: AbstractPEPS
    tn::Ansatz
    form::Form
end

Ansatz(tn::PEPS) = tn.tn

Base.copy(x::PEPS) = PEPS(copy(Ansatz(x)), form(x))
Base.similar(x::PEPS) = PEPS(similar(Ansatz(x)), form(x))
Base.zero(x::PEPS) = PEPS(zero(Ansatz(x)), form(x))

defaultorder(::Type{PEPS}) = (:o, :l, :r, :u, :d)
boundary(::PEPS) = Open()
form(tn::PEPS) = tn.form

# TODO periodic boundary conditions
# TODO non-square lattice
function PEPS(arrays::Matrix{<:AbstractArray}; order=defaultorder(PEPS))
    @assert ndims(arrays[1, 1]) == 3 "Array at (1,1) must have 3 dimensions"
    @assert ndims(arrays[1, end]) == 3 "Array at (1,end) must have 3 dimensions"
    @assert ndims(arrays[end, 1]) == 3 "Array at (end,1) must have 3 dimensions"
    @assert ndims(arrays[end, end]) == 3 "Array at (end,end) must have 3 dimensions"
    @assert all(
        ==(4) ∘ ndims,
        Iterators.flatten([
            arrays[1, 2:(end - 1)], arrays[end, 2:(end - 1)], arrays[2:(end - 1), 1], arrays[2:(end - 1), end]
        ]),
    ) "Arrays at boundaries must have 4 dimensions"
    @assert all(==(5) ∘ ndims, arrays[2:(end - 1), 2:(end - 1)]) "Inner arrays must have 5 dimensions"
    issetequal(order, defaultorder(PEPS)) ||
        throw(ArgumentError("order must be a permutation of $(String.(defaultorder(PEPS)))"))

    m, n = size(arrays)

    # predicate = all(eachindex(IndexCartesian(), arrays)) do I
    #     i, j = Tuple(I)
    #     array = arrays[i, j]

    #     N = ndims(array) - 1
    #     (i == 1 || i == m) && (N -= 1)
    #     (j == 1 || j == n) && (N -= 1)

    #     N > 0
    # end

    # if !predicate
    #     throw(DimensionMismatch())
    # end

    gen = IndexCounter()
    pinds = map(_ -> nextindex!(gen), arrays)
    vvinds = [nextindex!(gen) for _ in 1:(m - 1), _ in 1:n]
    hvinds = [nextindex!(gen) for _ in 1:m, _ in 1:(n - 1)]

    tn = TensorNetwork(
        map(eachindex(IndexCartesian(), arrays)) do I
            i, j = Tuple(I)

            array = arrays[i, j]
            pind = pinds[i, j]
            up = i == 1 ? missing : vvinds[i - 1, j]
            down = i == m ? missing : vvinds[i, j]
            left = j == 1 ? missing : hvinds[i, j - 1]
            right = j == n ? missing : hvinds[i, j]

            # TODO customize order
            Tensor(array, collect(skipmissing([pind, up, down, left, right])))
        end,
    )

    sitemap = Dict(Site(i, j) => pinds[i, j] for i in 1:m, j in 1:n)
    qtn = Quatum(tn, sitemap)
    graph = grid((m, n))
    # TODO fix this
    lattice = MetaGraph(graph, Site.(vertices(graph)) .=> nothing, map(x -> Site.(Tuple(x)) => nothing, edges(graph)))
    ansatz = Ansatz(qtn, lattice)
    return PEPS(ansatz, NonCanonical())
end

function Base.convert(::Type{PEPS}, tn::Product)
    @assert socket(tn) == State()

    # TODO fix this
    arrs::Matrix{<:AbstractArray} = arrays(tn)
    arrs[1] = reshape(arrs[1], size(arrs[1])..., 1)
    arrs[end] = reshape(arrs[end], size(arrs[end])..., 1)
    map!(@view(arrs[2:(end - 1)]), @view(arrs[2:(end - 1)])) do arr
        reshape(arr, size(arr)..., 1, 1)
    end

    return PEPS(arrs)
end

Base.adjoint(tn::PEPS) = PEPS(adjoint(Ansatz(tn)), form(tn))

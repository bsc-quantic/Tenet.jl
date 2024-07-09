struct Grid <: Ansatz
    super::Quantum
    boundary::Boundary
end

Base.copy(tn::Grid) = Grid(copy(Quantum(tn)), boundary(tn))

boundary(tn::Grid) = tn.boundary

PEPS(arrays) = Grid(State(), Open(), arrays)
pPEPS(arrays) = Grid(State(), Periodic(), arrays)
PEPO(arrays) = Grid(Operator(), Open(), arrays)
pPEPO(arrays) = Grid(Operator(), Periodic(), arrays)

alias(tn::Grid) = alias(socket(tn), boundary(tn), tn)
alias(::State, ::Open, ::Grid) = "PEPS"
alias(::State, ::Periodic, ::Grid) = "pPEPS"
alias(::Operator, ::Open, ::Grid) = "PEPO"
alias(::Operator, ::Periodic, ::Grid) = "pPEPO"

function Grid(::State, ::Periodic, arrays::Matrix{<:AbstractArray})
    @assert all(==(4) ∘ ndims, arrays) "All arrays must have 4 dimensions"

    m, n = size(arrays)
    gen = IndexCounter()
    pinds = map(_ -> nextindex(gen), arrays)
    hvinds = map(_ -> nextindex(gen), arrays)
    vvinds = map(_ -> nextindex(gen), arrays)

    _tensors = map(eachindex(IndexCartesian(), arrays)) do I
        i, j = Tuple(I)

        array = arrays[i, j]
        pind = pinds[i, j]
        up, down = hvinds[i, j], hvinds[mod1(i + 1, m), j]
        left, right = vvinds[i, j], vvinds[i, mod1(j + 1, n)]

        # TODO customize order
        Tensor(array, [pind, up, down, left, right])
    end

    sitemap = Dict(Site(i, j) => pinds[i, j] for i in 1:m, j in 1:n)

    return Grid(Quantum(TensorNetwork(_tensors), sitemap), Periodic())
end

function Grid(::State, ::Open, arrays::Matrix{<:AbstractArray})
    m, n = size(arrays)

    predicate = all(eachindex(arrays)) do I
        i, j = Tuple(I)
        array = arrays[i, j]

        N = ndims(array) - 1
        (i == 1 || i == m) && (N -= 1)
        (j == 1 || j == n) && (N -= 1)

        N > 0
    end

    if !predicate
        throw(DimensionMismatch())
    end

    gen = IndexCounter()
    pinds = map(_ -> nextindex(gen), arrays)
    vvinds = [nextindex(gen) for _ in 1:(m - 1), _ in 1:n]
    hvinds = [nextindex(gen) for _ in 1:m, _ in 1:(n - 1)]

    _tensors = map(eachindex(IndexCartesian(), arrays)) do I
        i, j = Tuple(I)

        array = arrays[i, j]
        pind = pinds[i, j]
        up = i == 1 ? missing : vvinds[i - 1, j]
        down = i == m ? missing : vvinds[i, j]
        left = j == 1 ? missing : hvinds[i, j - 1]
        right = j == n ? missing : hvinds[i, j]

        # TODO customize order
        Tensor(array, collect(skipmissing([pind, up, down, left, right])))
    end

    sitemap = Dict(Site(i, j) => pinds[i, j] for i in 1:m, j in 1:n)

    return Grid(Quantum(TensorNetwork(_tensors), sitemap), Open())
end

function Grid(::Operator, ::Periodic, arrays::Matrix{<:AbstractArray})
    @assert all(==(4) ∘ ndims, arrays) "All arrays must have 4 dimensions"

    m, n = size(arrays)
    gen = IndexCounter()
    ipinds = map(_ -> nextindex(gen), arrays)
    opinds = map(_ -> nextindex(gen), arrays)
    hvinds = map(_ -> nextindex(gen), arrays)
    vvinds = map(_ -> nextindex(gen), arrays)

    _tensors = map(eachindex(IndexCartesian(), arrays)) do I
        i, j = Tuple(I)

        array = arrays[i, j]
        ipind, opind = ipinds[i, j], opinds[i, j]
        up, down = hvinds[i, j], hvinds[mod1(i + 1, m), j]
        left, right = vvinds[i, j], vvinds[i, mod1(j + 1, n)]

        # TODO customize order
        Tensor(array, [ipind, opind, up, down, left, right])
    end

    sitemap = Dict(
        flatten([
            (Site(i, j; dual=true) => ipinds[i, j] for i in 1:m, j in 1:n),
            (Site(i, j) => opinds[i, j] for i in 1:m, j in 1:n),
        ]),
    )

    return Grid(Quantum(TensorNetwork(_tensors), sitemap), Periodic())
end

function Grid(::Operator, ::Open, arrays::Matrix{<:AbstractArray})
    m, n = size(arrays)

    predicate = all(eachindex(IndexCartesian(), arrays)) do I
        i, j = Tuple(I)
        array = arrays[i, j]

        N = ndims(array) - 2
        (i == 1 || i == m) && (N -= 1)
        (j == 1 || j == n) && (N -= 1)

        N > 0
    end

    if !predicate
        throw(DimensionMismatch())
    end

    gen = IndexCounter()
    ipinds = map(_ -> nextindex(gen), arrays)
    opinds = map(_ -> nextindex(gen), arrays)
    vvinds = [nextindex(gen) for _ in 1:(m - 1), _ in 1:n]
    hvinds = [nextindex(gen) for _ in 1:m, _ in 1:(n - 1)]

    _tensors = map(eachindex(IndexCartesian(), arrays)) do I
        i, j = Tuple(I)

        array = arrays[i, j]
        ipind = ipinds[i, j]
        opind = opinds[i, j]
        up = i == 1 ? missing : vvinds[i - 1, j]
        down = i == m ? missing : vvinds[i, j]
        left = j == 1 ? missing : hvinds[i, j - 1]
        right = j == n ? missing : hvinds[i, j]

        # TODO customize order
        Tensor(array, collect(skipmissing([ipind, opind, up, down, left, right])))
    end

    sitemap = Dict(
        flatten([
            (Site(i, j; dual=true) => ipinds[i, j] for i in 1:m, j in 1:n),
            (Site(i, j) => opinds[i, j] for i in 1:m, j in 1:n),
        ]),
    )

    return Grid(Quantum(TensorNetwork(_tensors), sitemap), Open())
end

function LinearAlgebra.transpose!(qtn::Grid)
    old = Quantum(qtn).sites
    new = Dict(Site(reverse(id(site)); dual=isdual(site)) => ind for (site, ind) in old)

    empty!(old)
    merge!(old, new)

    return qtn
end

Base.transpose(qtn::Grid) = LinearAlgebra.transpose!(copy(qtn))

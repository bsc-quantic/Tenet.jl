using Random

abstract type AbstractMPO <: AbstractAnsatz end

struct MPO <: AbstractAnsatz
    tn::Ansatz
    form::Form
end

Ansatz(tn::MPO) = tn.tn

Base.copy(x::MPO) = MPO(copy(Ansatz(x)), form(x))
Base.similar(x::MPO) = MPO(similar(Ansatz(x)), form(x))
Base.zero(x::MPO) = MPO(zero(Ansatz(x)), form(x))

defaultorder(::Type{MPO}) = (:o, :i, :l, :r)
boundary(::MPO) = Open()
form(tn::MPO) = tn.form

function MPO(arrays::Vector{<:AbstractArray}; order=defaultorder(MPO))
    @assert ndims(arrays[1]) == 3 "First array must have 3 dimensions"
    @assert all(==(4) ∘ ndims, arrays[2:(end - 1)]) "All arrays must have 4 dimensions"
    @assert ndims(arrays[end]) == 3 "Last array must have 3 dimensions"
    issetequal(order, defaultorder(MPO)) ||
        throw(ArgumentError("order must be a permutation of $(String.(defaultorder(MPO)))"))

    n = length(arrays)
    gen = IndexCounter()
    symbols = [nextindex!(gen) for _ in 1:(3n - 1)]

    tn = TensorNetwork(
        map(enumerate(arrays)) do (i, array)
            _order = if i == 1
                filter(x -> x != :l, order)
            elseif i == n
                filter(x -> x != :r, order)
            else
                order
            end

            inds = map(_order) do dir
                if dir == :o
                    symbols[i]
                elseif dir == :i
                    symbols[i + n]
                elseif dir == :l
                    symbols[2n + mod1(i - 1, n)]
                elseif dir == :r
                    symbols[2n + mod1(i, n)]
                else
                    throw(ArgumentError("Invalid direction: $dir"))
                end
            end
            Tensor(array, inds)
        end,
    )

    sitemap = Dict(Site(i) => symbols[i] for i in 1:n)
    merge!(sitemap, Dict(Site(i; dual=true) => symbols[i + n] for i in 1:n))
    qtn = Quantum(tn, sitemap)
    graph = path_graph(n)
    lattice = MetaGraph(graph, lanes(qtn) .=> nothing, map(x -> Site.(Tuple(x)) => nothing, edges(graph)))
    ansatz = Ansatz(qtn, lattice)
    return MPO(ansatz, NonCanonical())
end

function Base.convert(::Type{MPO}, tn::Product)
    @assert socket(tn) == Operator()

    arrs::Vector{Array} = arrays(tn)
    arrs[1] = reshape(arrs[1], size(arrs[1])..., 1)
    arrs[end] = reshape(arrs[end], size(arrs[end])..., 1)
    map!(@view(arrs[2:(end - 1)]), @view(arrs[2:(end - 1)])) do arr
        reshape(arr, size(arr)..., 1, 1)
    end

    return MPO(arrs)
end

Base.adjoint(tn::MPO) = MPO(adjoint(Ansatz(tn)), form(tn))

# TODO different input/output physical dims
# TODO let choose the orthogonality center
function Base.rand(rng::Random.AbstractRNG, ::Type{MPO}; n, maxdim, eltype=Float64, physdim=2)
    T = eltype
    ip = op = physdim
    χ = maxdim

    arrays::Vector{AbstractArray{T,N} where {N}} = map(1:n) do i
        χl, χr = let after_mid = i > n ÷ 2, i = (n + 1 - abs(2i - n - 1)) ÷ 2
            χl = min(χ, ip^(i - 1) * op^(i - 1))
            χr = min(χ, ip^i * op^i)

            # swap bond dims after mid and handle midpoint for odd-length MPS
            (isodd(n) && i == n ÷ 2 + 1) ? (χl, χl) : (after_mid ? (χr, χl) : (χl, χr))
        end

        # orthogonalize by QR factorization
        F = lq!(rand(rng, T, χl, ip * op * χr))
        reshape(Matrix(F.Q), χl, ip, op, χr)
    end

    # reshape boundary sites
    arrays[1] = reshape(arrays[1], ip, op, min(χ, ip * op))
    arrays[n] = reshape(arrays[n], min(χ, ip * op), ip, op)

    # TODO order might not be the best for performance
    return MPO(arrays; order=(:l, :i, :o, :r))
end

# TODO change it to `lanes`?
# TODO refactor common code with `MPS`
function sites(ψ::MPO, site::Site; dir)
    if dir === :left
        return site <= site"1" ? nothing : Site(id(site) - 1)
    elseif dir === :right
        return site >= Site(nlanes(ψ)) ? nothing : Site(id(site) + 1)
    else
        throw(ArgumentError("Unknown direction for MPO = :$dir"))
    end
end

@kwmethod function inds(ψ::MPO; at, dir)
    if dir === :left && at == site"1"
        return nothing
    elseif dir === :right && at == Site(nlanes(ψ); dual=isdual(at))
        return nothing
    elseif dir ∈ (:left, :right)
        return inds(ψ; bond=(at, sites(ψ, at; dir)))
    else
        throw(ArgumentError("Unknown direction for MPO = :$dir"))
    end
end

function evolve!(ψ::MPS, op::MPO; threshold=nothing, maxdim=nothing, renormalize=false) end
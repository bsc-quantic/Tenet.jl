using UUIDs: uuid4
using IterTools: partition
using Random

abstract type MatrixProductOperator{B} <: Operator{B} end

function MatrixProductOperator(arrays; bounds::Type{<:Bounds} = Open, kwargs...)
    MatrixProductOperator{bounds}(arrays; kwargs...)
end

function MatrixProductOperator{Open}(arrays; χ = nothing, order = (:l, :r, :i, :o), meta...)
    !issetequal(order, (:l, :r, :i, :o)) &&
        throw(ArgumentError("`order` must be a permutation of the :l, :r, :i and :o"))
    order = Dict(dim => i for (i, dim) in enumerate(order))

    # check format
    # NOTE matching index dimensions are checked on `push!(::TensorNetwork,...)`
    # TODO check χ?
    if !all(==(4) ∘ ndims, arrays[2:end-1]) || ndims(first(arrays)) != 3 || ndims(last(arrays)) != 3
        throw(DimensionMismatch("virtual bond-dims mismatch"))
    end

    tn = TensorNetwork{MatrixProductOperator{Open}}(; χ = χ, order = order, meta...)

    n = length(arrays)
    vinds = Dict(x => Symbol(uuid4()) for x in partition(1:n, 2, 1))
    oinds = Dict(i => Symbol(uuid4()) for i in 1:n)
    iinds = Dict(i => Symbol(uuid4()) for i in 1:n)

    for (i, data) in enumerate(arrays)
        # Handle boundary cases and inner tensors separately
        if i == 1
            labels = [vinds[(1, 2)], iinds[1], oinds[1]]
            original_order = [:r, :i, :o]
        elseif i == n
            labels = [vinds[(n - 1, n)], iinds[n], oinds[n]]
            original_order = [:l, :i, :o]
        else
            lind = vinds[(i - 1, i)]
            rind = vinds[(i, i + 1)]
            labels = [lind, rind, iinds[i], oinds[i]]
            original_order = [:l, :r, :i, :o]
        end

        # Filter the order dictionary based on the original_order and sort it following order
        filtered_order = [p[1] for p in sort(filter(p -> p.first ∈ original_order, collect(order)), by = x -> x[2])]
        permutator = (ind -> Dict(p => i for (i, p) in enumerate(filtered_order))[ind]).(original_order)

        invpermute!(labels, permutator)
        alias = Dict([x => y for (x, y) in zip(invpermute!(original_order, permutator), labels)])
        push!(tn, Tensor(data, labels; alias = alias))
    end

    # mark input indices
    for (site, label) in iinds
        index = inds(tn, label)
        index.meta[:site] = site
        index.meta[:plug] = :input
    end
    # mark output indices
    for (site, label) in oinds
        index = inds(tn, label)
        index.meta[:site] = site
        index.meta[:plug] = :output
    end

    return tn
end

function MatrixProductOperator{Closed}(arrays; χ = nothing, order = (:l, :r, :i, :o), meta...)
    !issetequal(order, (:l, :r, :i, :o)) &&
        throw(ArgumentError("`order` must be a permutation of the :l, :r, :i and :o"))
    order = Dict(side => i for (i, side) in enumerate(order))

    # check format
    # TODO check χ?
    if !all(==(4) ∘ ndims, arrays)
        throw(DimensionMismatch("virtual bond-dims mismatch"))
    end

    tn = TensorNetwork{MatrixProductOperator{Closed}}(; χ = χ, order = order, meta...)

    n = length(arrays)
    vinds = Dict(x => Symbol(uuid4()) for x in ringpeek(1:n))
    oinds = Dict(i => Symbol(uuid4()) for i in 1:n)
    iinds = Dict(i => Symbol(uuid4()) for i in 1:n)
    permutator = [order[i] for i in (:l, :r, :i, :o)]

    # add tensors
    for (i, data) in enumerate(arrays)
        lind = vinds[(mod1(i - 1, n), i)]
        rind = vinds[(i, mod1(i + 1, n))]

        labels = [lind, rind, iinds[i], oinds[i]]
        invpermute!(labels, permutator)
        alias = Dict([x => y for (x, y) in zip(invpermute!([:l, :r, :i, :o], permutator), labels)])

        tensor = Tensor(data, labels; alias = alias)
        push!(tn, tensor)
    end

    # mark input indices
    for (site, label) in iinds
        index = inds(tn, label)
        index.meta[:site] = site
        index.meta[:plug] = :input
    end
    # mark output indices
    for (site, label) in oinds
        index = inds(tn, label)
        index.meta[:site] = site
        index.meta[:plug] = :output
    end

    return tn
end

struct MPOSampler{B<:Bounds,T} <: Random.Sampler{TensorNetwork{MatrixProductOperator{B}}}
    n::Int
    i::Int
    o::Int
    χ::Int
end

Base.eltype(::MPOSampler{B}) where {B<:Bounds} = TensorNetwork{MatrixProductOperator{B}}

function Base.rand(
    ::Type{MatrixProductOperator{B}},
    n::Integer,
    i::Integer,
    o::Integer,
    χ::Integer;
    eltype::Type = Float64,
) where {B<:Bounds}
    rand(MPOSampler{B,eltype}(n, i, o, χ))
end

Base.rand(::Type{MatrixProductOperator}, args::Vararg{Integer,3}; kwargs...) =
    rand(MatrixProductOperator{Open}, args...; kwargs...)

# TODO let choose the orthogonality center
function Base.rand(rng::Random.AbstractRNG, sampler::MPOSampler{Open,T}) where {T}
    n, χ, ip, op = getfield.((sampler,), (:n, :χ, :i, :o))

    arrays::Vector{AbstractArray{T,N} where {N}} = map(1:n) do i
        χl, χr = let after_mid = i > n ÷ 2, i = (n + 1 - abs(2i - n - 1)) ÷ 2
            χl = min(χ, ip^(i - 1) * op^(i - 1))
            χr = min(χ, ip^i * op^i)

            # swap bond dims after mid
            after_mid ? (χr, χl) : (χl, χr)
        end

        shape = if i == 1
            (χr, ip, op)
        elseif i == n
            (χl, ip, op)
        else
            (χl, χr, ip, op)
        end

        # orthogonalize by Gram-Schmidt algorithm
        A = gramschmidt!(rand(rng, T, shape[1], prod(shape[2:end])))
        reshape(A, shape)
    end

    # normalize
    ζ = min(χ, ip * op)
    arrays[1] ./= sqrt(ζ)

    MatrixProductOperator{Open}(arrays; χ = χ)
end

# TODO stable renormalization
function Base.rand(rng::Random.AbstractRNG, sampler::MPOSampler{Closed,T}) where {T}
    n, χ, i, o = getfield.((sampler,), (:n, :χ, :i, :o))
    MatrixProductOperator{Closed}([rand(rng, T, n, χ, i, o) for _ in 1:n]; χ = χ)
end

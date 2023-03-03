using OptimizedEinsum: ContractionPath
using UUIDs: uuid4
using Base.Iterators: flatten
using IterTools: partition
using Random

abstract type MatrixProductState{B} <: State{B} end

function MatrixProductState(arrays; bounds::Type{<:Bounds} = Open, kwargs...)
    MatrixProductState{bounds}(arrays; kwargs...)
end

function MatrixProductState{Open}(arrays; χ = nothing, order = (:l, :r, :p), meta...)
    !issetequal(order, (:l, :r, :p)) && throw(ArgumentError("`order` must be a permutation of the :l, :r and :p"))
    order = Dict(dim => i for (i, dim) in enumerate(order))

    # check format
    # NOTE matching index dimensions are checked on `push!(::TensorNetwork,...)`
    # TODO check χ?
    if !all(==(3) ∘ ndims, arrays[2:end-1]) || ndims(first(arrays)) != 2 || ndims(last(arrays)) != 2
        throw(DimensionMismatch("virtual bond-dims mismatch"))
    end

    tn = TensorNetwork{MatrixProductState{Open}}(; χ = χ, order = order, meta...)

    n = length(arrays)
    vinds = Dict(x => Symbol(uuid4()) for x in partition(1:n, 2, 1))
    pinds = Dict(i => Symbol(uuid4()) for i in 1:n)
    permutator = [order[i] for i in (:l, :r, :p)]

    # add boundary tensors
    labels = if order[:r] < order[:p]
        (vinds[(1, 2)], pinds[1])
    else
        (pinds[1], vinds[(1, 2)])
    end
    push!(tn, Tensor(first(arrays), labels))

    labels = if order[:l] < order[:p]
        (vinds[(n - 1, n)], pinds[n])
    else
        (pinds[n], vinds[(n - 1, n)])
    end
    push!(tn, Tensor(last(arrays), labels))

    # add tensors
    for (i, data) in zip(2:n-1, arrays[2:end-1])
        lind = vinds[(mod1(i - 1, n), i)]
        rind = vinds[(i, mod1(i + 1, n))]

        labels = [lind, rind, pinds[i]]
        permute!(labels, permutator)

        tensor = Tensor(data, labels)
        push!(tn, tensor)
    end

    # mark physical indices
    for (site, label) in pinds
        index = inds(tn, label)
        index.meta[:site] = site
        index.meta[:plug] = :output
    end

    return tn
end

function MatrixProductState{Closed}(arrays; χ = nothing, order = (:l, :r, :p), meta...)
    !issetequal(order, (:l, :r, :p)) && throw(ArgumentError("`order` must be a permutation of the :l, :r and :p"))
    order = Dict(side => i for (i, side) in enumerate(order))

    # check format
    # TODO check χ?
    if !all(==(3) ∘ ndims, arrays)
        throw(DimensionMismatch("virtual bond-dims mismatch"))
    end

    tn = TensorNetwork{MatrixProductState{Closed}}(; χ = χ, order = order, meta...)

    n = length(arrays)
    vinds = Dict(x => Symbol(uuid4()) for x in ringpeek(1:n))
    pinds = Dict(i => Symbol(uuid4()) for i in 1:n)
    permutator = [order[i] for i in (:l, :r, :p)]

    # add tensors
    for (i, data) in enumerate(arrays)
        lind = vinds[(mod1(i - 1, n), i)]
        rind = vinds[(i, mod1(i + 1, n))]

        labels = [lind, rind, pinds[i]]
        permute!(labels, permutator)

        tensor = Tensor(data, labels)
        push!(tn, tensor)
    end

    # mark physical indices
    for (site, label) in pinds
        index = inds(tn, label)
        index.meta[:site] = site
        index.meta[:plug] = :output
    end

    return tn
end

# NOTE does not use optimal contraction path, but "parallel-optimal" which costs x2 more
# function contractpath(a::TensorNetwork{<:MatrixProductState}, b::TensorNetwork{<:MatrixProductState})
#     !issetequal(sites(a), sites(b)) && throw(ArgumentError("both tensor networks are expected to have same sites"))

#     b = replace(b, [nameof(outsiteind(b, s)) => nameof(outsiteind(a, s)) for s in sites(a)]...)
#     path = nameof.(flatten([physicalinds(a), flatten(zip(virtualinds(a), virtualinds(b)))]) |> collect)
#     inputs = flatten([tensors(a), tensors(b)]) .|> labels
#     output = Symbol[]
#     size_dict = merge(size(a), size(b))

#     ContractionPath(path, inputs, output, size_dict)
# end

# Base.push!(::TensorNetwork{MatrixProductState}, args...; kwargs...) =
#     throw(MethodError("push! is forbidden for MatrixProductState"))

# Base.pop!(::TensorNetwork{MatrixProductState}, args...; kwargs...) =
#     throw(MethodError("pop! is forbidden for MatrixProductState"))

struct MPSSampler{B<:Bounds,T} <: Random.Sampler{TensorNetwork{MatrixProductState{B}}}
    n::Int
    p::Int
    χ::Int
end

Base.eltype(::MPSSampler{B}) where {B<:Bounds} = TensorNetwork{MatrixProductState{B}}

function Base.rand(
    ::Type{MatrixProductState{B}},
    n::Integer,
    p::Integer,
    χ::Integer;
    eltype::Type = Float64,
) where {B<:Bounds}
    rand(MPSSampler{B,eltype}(n, p, χ))
end

# NOTE `Vararg{Integer,3}` for dissambiguation
Base.rand(::Type{MatrixProductState}, args::Vararg{Integer,3}; kwargs...) =
    rand(MatrixProductState{Open}, args...; kwargs...)

function Base.rand(rng::Random.AbstractRNG, sampler::MPSSampler{Open,T}) where {T}
    n, χ, p = getfield.((sampler,), (:n, :χ, :p))

    arrays::Vector{AbstractArray{T,N} where {N}} = map(1:n) do i
        let i = (n + 1 - abs(2i - n - 1)) ÷ 2
            χl = min(χ, p^(i - 1))
            χr = min(χ, p^i)
        end

        # swap bond dims after mid
        i > n ÷ 2 && ((χl, χr) = (χr, χl))

        # orthogonalize by solving linear systems
        A = rand(rng, T, χl, χr * p)
        for i in 1:χl-1
            M = A[1:i, 1:i]
            B = [-dot(A[j, i+1:end], A[i+1, i+1:end]) for j in 1:i]
            X = M \ B
            A[i+1, 1:i] = X
        end

        # normalize rows
        foreach(row -> normalize!(row), eachrow(A))

        reshape(A, χl, χr, p)
    end

    arrays[1] = reshape(arrays[1], p, p)
    arrays[n] = reshape(arrays[n], p, p)

    MatrixProductState{Open}(arrays; χ = χ)
end

# TODO stable renormalization
function Base.rand(rng::Random.AbstractRNG, sampler::MPSSampler{Closed,T}) where {T}
    n, χ, p = getfield.((sampler,), (:n, :χ, :p))
    ψ = MatrixProductState{Closed}([rand(rng, T, χ, χ, p) for _ in 1:n]; χ = χ)
    normalize!(ψ)

    return ψ
end

using UUIDs: uuid4
using Base.Iterators: flatten
using IterTools: partition
using Random

abstract type MatrixProduct{P,B} <: Quantum where {P<:Plug,B<:Boundary} end

boundary(::Type{<:MatrixProduct{P,B}}) where {P,B} = B
plug(::Type{<:MatrixProduct{P}}) where {P} = P

function MatrixProduct{P}(arrays; boundary::Type{<:Boundary} = Open, kwargs...) where {P<:Plug}
    MatrixProduct{P,boundary}(arrays; kwargs...)
end

function checkmeta(::Type{MatrixProduct{P,B}}, tn::TensorNetwork) where {P,B}
    # meta exists
    haskey(tn.metadata, :χ) || return false

    # meta has correct type
    tn[:χ] isa Integer && tn[:χ] > 0 || isnothing(tn[:χ]) || return false

    # no virtual index has dimensionality bigger than χ
    all(i -> size(tn, i) <= tn[:χ], labels(tn, :inner)) || return false

    return true
end

_sitealias(::Type{MatrixProduct{P,Open}}, order, n, i) where {P<:Plug} =
    if i == 1
        filter(!=(:l), order)
    elseif i == n
        filter(!=(:r), order)
    else
        order
    end
_sitealias(::Type{MatrixProduct{P,Periodic}}, order, n, i) where {P<:Plug} = tuple(order...)

defaultorder(::Type{MatrixProduct{State}}) = (:l, :r, :o)
defaultorder(::Type{MatrixProduct{Operator}}) = (:l, :r, :i, :o)

function MatrixProduct{P,B}(
    arrays;
    χ = nothing,
    order = defaultorder(MatrixProduct{P}),
    metadata...,
) where {P<:Plug,B<:Boundary}
    issetequal(order, defaultorder(MatrixProduct{P})) || throw(
        ArgumentError(
            "`order` must be a permutation of $(join(String.(defaultorder(MatrixProduct{P})), ',', " and "))",
        ),
    )

    n = length(arrays)
    vinds = Dict(x => Symbol(uuid4()) for x in ringpeek(1:n))
    oinds = Dict(i => Symbol(uuid4()) for i in 1:n)
    iinds = Dict(i => Symbol(uuid4()) for i in 1:n)

    # mark plug connectors
    plug = Dict((site, :out) => label for (site, label) in oinds)

    tensors = map(enumerate(arrays)) do (i, array)
        dirs = _sitealias(MatrixProduct{P,B}, order, n, i)

        labels = map(dirs) do dir
            if dir === :l
                vinds[(mod1(i - 1, n), i)]
            elseif dir === :r
                vinds[(i, mod1(i + 1, n))]
            elseif dir === :o
                oinds[i]
            elseif dir === :i
                iinds[i]
            end
        end
        alias = Dict(dir => label for (dir, label) in zip(dirs, labels))

        Tensor(array, labels; alias = alias)
    end

    return TensorNetwork{MatrixProduct{P,B}}(tensors; χ, plug, metadata...)
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

# TODO let choose the orthogonality center
function Base.rand(rng::Random.AbstractRNG, sampler::TNSampler{MatrixProduct{State,Open}})
    n = sampler.n
    χ = sampler.χ
    p = sampler.p
    T = get(sampler, :eltype, Float64)

    arrays::Vector{AbstractArray{T,N} where {N}} = map(1:n) do i
        χl, χr = let after_mid = i > n ÷ 2, i = (n + 1 - abs(2i - n - 1)) ÷ 2
            χl = min(χ, p^(i - 1))
            χr = min(χ, p^i)

            # swap bond dims after mid and handle midpoint for odd-length MPS
            (isodd(n) && i == n ÷ 2 + 1) ? (χl, χl) : (after_mid ? (χr, χl) : (χl, χr))
        end

        # fix for first site
        i == 1 && ((χl, χr) = (χr, 1))

        # orthogonalize by Gram-Schmidt algorithm
        A = gramschmidt!(rand(rng, T, χl, χr * p))

        reshape(A, χl, χr, p)
    end

    # reshape boundary sites
    arrays[1] = reshape(arrays[1], p, p)
    arrays[n] = reshape(arrays[n], p, p)

    # normalize state
    arrays[1] ./= sqrt(p)

    MatrixProduct{State,Open}(arrays; χ = χ)
end

# TODO let choose the orthogonality center
# TODO different input/output physical dims
function Base.rand(rng::Random.AbstractRNG, sampler::TNSampler{MatrixProduct{Operator,Open}})
    n = sampler.n
    χ = sampler.χ
    p = sampler.p
    T = get(sampler, :eltype, Float64)

    ip = op = p

    arrays::Vector{AbstractArray{T,N} where {N}} = map(1:n) do i
        χl, χr = let after_mid = i > n ÷ 2, i = (n + 1 - abs(2i - n - 1)) ÷ 2
            χl = min(χ, ip^(i - 1) * op^(i - 1))
            χr = min(χ, ip^i * op^i)

            # swap bond dims after mid and handle midpoint for odd-length MPS
            (isodd(n) && i == n ÷ 2 + 1) ? (χl, χl) : (after_mid ? (χr, χl) : (χl, χr))
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
# TODO different input/output physical dims for Operator
function Base.rand(rng::Random.AbstractRNG, sampler::TNSampler{MatrixProduct{P,Periodic}}) where {P<:Plug}
    n = sampler.n
    χ = sampler.χ
    p = sampler.p
    T = get(sampler, :eltype, Float64)

    A = MatrixProduct{P,Periodic}([rand(rng, T, [P === State ? (χ, χ, p) : (χ, χ, p, p)]...) for _ in 1:n]; χ = χ)
    normalize!(A)

    return A
end

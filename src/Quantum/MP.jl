using UUIDs: uuid4
using Base.Iterators: flatten
using Random
using Bijections
using Muscle: gramschmidt!
using EinExprs: inds

"""
    MatrixProduct{P<:Plug,B<:Boundary} <: Quantum

A generic ansatz representing Matrix Product State (MPS) and Matrix Product Operator (MPO) topology, aka Tensor Train.
Type variable `P` represents the `Plug` type (`State` or `Operator`) and `B` represents the `Boundary` type (`Open` or `Periodic`).

# Ansatz Fields

  - `χ::Union{Nothing,Int}` Maximum virtual bond dimension.
"""
abstract type MatrixProduct{P,B} <: Quantum where {P<:Plug,B<:Boundary} end

boundary(::Type{<:MatrixProduct{P,B}}) where {P,B} = B
plug(::Type{<:MatrixProduct{P}}) where {P} = P

function MatrixProduct{P}(arrays; boundary::Type{<:Boundary} = Open, kwargs...) where {P<:Plug}
    MatrixProduct{P,boundary}(arrays; kwargs...)
end

metadata(::Type{<:MatrixProduct}) = merge(metadata(supertype(MatrixProduct)), @NamedTuple begin
    χ::Union{Nothing,Int}
end)

function checkmeta(::Type{MatrixProduct{P,B}}, tn::TensorNetwork) where {P,B}
    # meta has correct type
    isnothing(tn.χ) || tn.χ > 0 || return false

    # no virtual index has dimensionality bigger than χ
    all(i -> isnothing(tn.χ) || size(tn, i) <= tn.χ, inds(tn, :virtual)) || return false

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
_sitealias(::Type{MatrixProduct{P,Infinite}}, order, n, i) where {P<:Plug} = tuple(order...)

defaultorder(::Type{MatrixProduct{State}}) = (:l, :r, :o)
defaultorder(::Type{MatrixProduct{Operator}}) = (:l, :r, :i, :o)

"""
    MatrixProduct{P,B}(arrays::AbstractArray[]; χ::Union{Nothing,Int} = nothing, order = defaultorder(MatrixProduct{P}))

Construct a [`TensorNetwork`](@ref) with [`MatrixProduct`](@ref) ansatz, from the arrays of the tensors.

# Keyword Arguments

  - `χ` Maximum virtual bond dimension. Defaults to `nothing`.
  - `order` Order of tensor indices on `arrays`. Defaults to `(:l, :r, :o)` if `P` is a `State`, `(:l, :r, :i, :o)` if `Operator`.
"""
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

    interlayer = if P <: State
        [Bijection(oinds)]
    elseif P <: Operator
        [Bijection(iinds), Bijection(oinds)]
    else
        throw(ErrorException("Plug $P is not valid"))
    end

    tensors = map(enumerate(arrays)) do (i, array)
        dirs = _sitealias(MatrixProduct{P,B}, order, n, i)

        inds = map(dirs) do dir
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
        alias = Dict(dir => label for (dir, label) in zip(dirs, inds))

        Tensor(array, inds; alias = alias)
    end

    return TensorNetwork{MatrixProduct{P,B}}(tensors; χ, plug = P, interlayer, metadata...)
end

const MPS = MatrixProduct{State}
const MPO = MatrixProduct{Operator}

tensors(ψ::TensorNetwork{MatrixProduct{P,Infinite}}, site::Int, args...) where {P<:Plug} =
    tensors(plug(ψ), ψ, mod1(site, length(ψ.tensors)), args...)

Base.length(ψ::TensorNetwork{MatrixProduct{P,Infinite}}) where {P<:Plug} = Inf

# NOTE does not use optimal contraction path, but "parallel-optimal" which costs x2 more
# function contractpath(a::TensorNetwork{<:MatrixProductState}, b::TensorNetwork{<:MatrixProductState})
#     !issetequal(sites(a), sites(b)) && throw(ArgumentError("both tensor networks are expected to have same sites"))

#     b = replace(b, [nameof(outsiteind(b, s)) => nameof(outsiteind(a, s)) for s in sites(a)]...)
#     path = nameof.(flatten([physicalinds(a), flatten(zip(virtualinds(a), virtualinds(b)))]) |> collect)
#     inputs = flatten([tensors(a), tensors(b)]) .|> inds
#     output = Symbol[]
#     size_dict = merge(size(a), size(b))

#     ContractionPath(path, inputs, output, size_dict)
# end

# TODO let choose the orthogonality center
function Base.rand(rng::Random.AbstractRNG, sampler::TNSampler{MatrixProduct{State,Open}})
    n = sampler.n
    χ = sampler.χ
    p = get(sampler, :p, 2)
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
    p = get(sampler, :p, 2)
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

    MatrixProduct{Operator,Open}(arrays; χ = χ)
end

# TODO stable renormalization
# TODO different input/output physical dims for Operator
function Base.rand(rng::Random.AbstractRNG, sampler::TNSampler{MatrixProduct{P,Periodic}}) where {P<:Plug}
    n = sampler.n
    χ = sampler.χ
    p = get(sampler, :p, 2)
    T = get(sampler, :eltype, Float64)

    A = MatrixProduct{P,Periodic}([rand(rng, T, [P === State ? (χ, χ, p) : (χ, χ, p, p)]...) for _ in 1:n]; χ = χ)
    normalize!(A)

    return A
end

using UUIDs: uuid4
using Base.Iterators: flatten
using Random
using Bijections: Bijection
using Muscle: gramschmidt!
using EinExprs: inds

"""
    MatrixProduct{S,B} <: Ansatz{S,B}

A generic ansatz representing Matrix Product State (MPS) and Matrix Product Operator (MPO) topology, aka Tensor Train.
"""
struct MatrixProduct{S,B} <: Ansatz{S,B} end
(T::Type{MatrixProduct{S}})(arrays; boundary::Boundary = Open(), kwargs...) where {S} = T{boundary}(arrays; kwargs...)

const MPS = MatrixProduct{State}
const MPO = MatrixProduct{Operator}

_sitealias(::Type{MatrixProduct{S,Open}}, order, n, i) where {S} =
    if i == 1
        filter(!=(:l), order)
    elseif i == n
        filter(!=(:r), order)
    else
        order
    end
_sitealias(::Type{MatrixProduct{S,Periodic}}, order, n, i) where {S} = tuple(order...)
_sitealias(::Type{MatrixProduct{S,Infinite}}, order, n, i) where {S} = tuple(order...)

defaultorder(::Type{MatrixProduct{State}}) = (:l, :r, :o)
defaultorder(::Type{MatrixProduct{Operator}}) = (:l, :r, :i, :o)

"""
    MatrixProduct{S,B}(arrays::AbstractArray[]; order = defaultorder(MatrixProduct{S})) where {S<:Socket, B<:Boundary}

Construct a [`TensorNetwork`](@ref) with [`MatrixProduct`](@ref) ansatz, from the arrays of the tensors.

# Keyword Arguments

  - `order` Order of tensor indices on `arrays`. Defaults to `(:l, :r, :o)` if `P` is a `State`, `(:l, :r, :i, :o)` if `Operator`.
"""
function (A::Type{MatrixProduct{S,B}})(arrays; order = defaultorder(MatrixProduct{S})) where {S,B}
    issetequal(order, defaultorder(MatrixProduct{S})) || throw(
        ArgumentError(
            "`order` must be a permutation of $(join(String.(defaultorder(MatrixProduct{S})), ',', " and "))",
        ),
    )

    n = length(arrays)
    vinds = Dict(x => Symbol(uuid4()) for x in ringpeek(1:n))
    oinds = Dict(i => Symbol(uuid4()) for i in 1:n)
    iinds = Dict(i => Symbol(uuid4()) for i in 1:n)

    plug = if S <: State
        [Bijection(oinds)]
    elseif S <: Operator
        [Bijection(iinds), Bijection(oinds)]
    else
        throw(ErrorException("Socket $S is not valid"))
    end

    if B <: Open
        filter!(splat(<) ∘ first, vinds)
    end

    tensors = map(enumerate(arrays)) do (i, array)
        dirs = _sitealias(A, order, n, i)

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

        Tensor(array, inds)
    end

    return TensorNetwork{Quantum}(tensors; layer = [Set(values(vinds))], plug, ansatz = [A()])
end

tensors(ψ::TensorNetwork{MatrixProduct{S,Infinite}}, site::Int, args...) where {S<:Socket} =
    tensors(plug(ψ), ψ, mod1(site, length(ψ.tensors)), args...)

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
function Base.rand(rng::Random.AbstractRNG, A::Type{MatrixProduct{State,Open}}; n, χ, p = 2, eltype = Float64)
    arrays::Vector{AbstractArray{eltype,N} where {N}} = map(1:n) do i
        χl, χr = let after_mid = i > n ÷ 2, i = (n + 1 - abs(2i - n - 1)) ÷ 2
            χl = min(χ, p^(i - 1))
            χr = min(χ, p^i)

            # swap bond dims after mid and handle midpoint for odd-length MPS
            (isodd(n) && i == n ÷ 2 + 1) ? (χl, χl) : (after_mid ? (χr, χl) : (χl, χr))
        end

        # fix for first site
        i == 1 && ((χl, χr) = (χr, 1))

        # orthogonalize by Gram-Schmidt algorithm
        array = gramschmidt!(rand(rng, eltype, χl, χr * p))
        reshape(array, χl, χr, p)
    end

    # reshape boundary sites
    arrays[1] = reshape(arrays[1], p, p)
    arrays[n] = reshape(arrays[n], p, p)

    # normalize state
    arrays[1] ./= sqrt(p)

    A(arrays)
end

# TODO let choose the orthogonality center
# TODO different input/output physical dims
function Base.rand(rng::Random.AbstractRNG, A::Type{MatrixProduct{Operator,Open}}; n, χ, p = 2, eltype = Float64)
    ip = op = p

    arrays::Vector{AbstractArray{eltype,N} where {N}} = map(1:n) do i
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
        array = gramschmidt!(rand(rng, eltype, shape[1], prod(shape[2:end])))
        reshape(array, shape)
    end

    # normalize
    ζ = min(χ, ip * op)
    arrays[1] ./= sqrt(ζ)

    A(arrays)
end

# TODO stable renormalization
# TODO different input/output physical dims for Operator
function Base.rand(
    rng::Random.AbstractRNG,
    A::Type{MatrixProduct{S,Periodic}};
    n,
    χ,
    p = 2,
    eltype = Float64,
) where {S<:Socket}
    arrays = [rand(rng, eltype, [S === State ? (χ, χ, p) : (χ, χ, p, p)]...) for _ in 1:n]
    ψ = A(arrays)
    # normalize!(ψ)
    return ψ
end

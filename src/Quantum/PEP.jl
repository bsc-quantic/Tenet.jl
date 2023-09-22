using UUIDs: uuid4
using EinExprs: inds

"""
    ProjectedEntangledPair{S,B} <: Ansatz{S,B}

A generic ansatz representing Projected Entangled Pair States (PEPS) and Projected Entangled Pair Operators (PEPO).
"""
struct ProjectedEntangledPair{S,B} <: Ansatz{S,B} end
(T::Type{ProjectedEntangledPair{S}})(arrays; boundary::Boundary = Open(), kwargs...) where {S} =
    T{boundary}(arrays; kwargs...)

const PEPS = ProjectedEntangledPair{State}
const PEPO = ProjectedEntangledPair{Operator}

function _sitealias(::Type{ProjectedEntangledPair{S,Open}}, order, size, pos) where {S<:Socket}
    m, n = size
    i, j = pos

    order = [order...]

    filter(order) do dir
        !(i == 1 && dir === :u || i == m && dir === :d || j == 1 && dir === :l || j == n && dir === :r)
    end
end
_sitealias(::Type{ProjectedEntangledPair{S,Periodic}}, order, _, _) where {S<:Socket} = tuple(order...)
_sitealias(::Type{ProjectedEntangledPair{S,Infinite}}, order, _, _) where {S<:Socket} = tuple(order...)

defaultorder(::Type{ProjectedEntangledPair{State}}) = (:l, :r, :u, :d, :o)
defaultorder(::Type{ProjectedEntangledPair{Operator}}) = (:l, :r, :u, :d, :i, :o)

"""
    ProjectedEntangledPair{S,B}(arrays::Matrix{AbstractArray}; order = defaultorder(ProjectedEntangledPair{S}))

Construct a [`TensorNetwork`](@ref) with [`ProjectedEntangledPair`](@ref) ansatz, from the arrays of the tensors.

# Keyword Arguments

  - `order` Order of the tensor indices on `arrays`. Defaults to `(:l, :r, :u, :d, :o)` if `P` is a `State`, `(:l, :r, :u, :d, :i, :o)` if `Operator`.
"""
function (A::Type{ProjectedEntangledPair{S,B}})(
    arrays;
    order = defaultorder(ProjectedEntangledPair{S}),
) where {S<:Socket,B<:Boundary}
    issetequal(order, defaultorder(ProjectedEntangledPair{S})) || throw(
        ArgumentError(
            "`order` must be a permutation of $(join(String.(defaultorder(ProjectedEntangledPair{S})), ',', " and "))",
        ),
    )

    m, n = size(arrays)
    hinds = Dict((i, j) => Symbol(uuid4()) for i in 1:m, j in ringpeek(1:n))
    vinds = Dict((i, j) => Symbol(uuid4()) for i in ringpeek(1:m), j in 1:n)
    oinds = Dict((i, j) => Symbol(uuid4()) for i in 1:m, j in 1:n)
    iinds = Dict((i, j) => Symbol(uuid4()) for i in 1:m, j in 1:n)

    plug = if S <: State
        [Bijection(Dict(i + j * m => index for ((i, j), index) in oinds))]
    elseif S <: Operator
        [
            Bijection(Dict(i + j * m => index for ((i, j), index) in iinds)),
            Bijection(Dict(i + j * m => index for ((i, j), index) in oinds)),
        ]
    else
        throw(ErrorException("Socket $S is not valid"))
    end

    if B <: Open
        filter!(splat(<) ∘ first ∘ first, vinds)
    end

    tensors = map(zip(Iterators.map(Tuple, eachindex(IndexCartesian(), arrays)), arrays)) do ((i, j), array)
        dirs = _sitealias(ProjectedEntangledPair{S,B}, order, (m, n), (i, j))

        inds = map(dirs) do dir
            if dir === :l
                hinds[(i, (mod1(j - 1, n), j))]
            elseif dir === :r
                hinds[(i, (j, mod1(j + 1, n)))]
            elseif dir === :u
                vinds[((mod1(i - 1, m), i), j)]
            elseif dir === :d
                vinds[((i, mod1(i + 1, m)), j)]
            elseif dir === :i
                iinds[(i, j)]
            elseif dir === :o
                oinds[(i, j)]
            end
        end

        Tensor(array, inds)
    end |> vec

    return TensorNetwork{Quantum}(tensors; layer = [Set(values(vinds))], plug, ansatz = [A()])
end

tensors(ψ::TensorNetwork{ProjectedEntangledPair{S,Infinite}}, site::Int, args...) where {S<:Socket} =
    tensors(plug(ψ), ψ, mod1(site, length(ψ.tensors)), args...)

# TODO normalize
# TODO let choose the orthogonality center
# TODO different input/output physical dims
function Base.rand(
    rng::Random.AbstractRNG,
    A::Type{ProjectedEntangledPair{S,Open}};
    rows,
    cols,
    χ,
    p = 2,
    eltype = Float64,
) where {S<:Socket}
    arrays::Matrix{AbstractArray{eltype,N} where {N}} = reshape(
        map(Iterators.product(1:rows, 1:cols)) do (i, j)
            shape = filter(
                !=(1),
                [
                    i === 1 ? 1 : χ,
                    i === rows ? 1 : χ,
                    j === 1 ? 1 : χ,
                    j === cols ? 1 : χ,
                    p,
                    if S <: State
                        1
                    elseif S <: Operator
                        p
                    else
                        throw(ErrorException("$S is not a valid Socket type"))
                    end,
                ],
            )

            rand(rng, eltype, shape...)
        end,
        rows,
        cols,
    )

    # normalize state
    arrays[1, 1] ./= S <: State ? sqrt(p) : p

    A(arrays)
end

# TODO normalize
# TODO let choose the orthogonality center
# TODO different input/output physical dims
function Base.rand(
    rng::Random.AbstractRNG,
    A::Type{ProjectedEntangledPair{S,Periodic}};
    rows,
    cols,
    χ,
    p = 2,
    eltype = Float64,
) where {S<:Socket}
    arrays::Matrix{AbstractArray{eltype,N} where {N}} = reshape(
        map(Iterators.product(1:rows, 1:cols)) do (i, j)
            shape = tuple([χ, χ, χ, χ]..., ([if S <: State
                (p,)
            elseif S <: Operator
                (p, p)
            else
                throw(ErrorException("$S is not a valid Socket type"))
            end]...)...)

            # A = gramschmidt!(rand(rng, T, shape[1], prod(shape[1:end])))
            rand(rng, eltype, shape...)
        end,
        rows,
        cols,
    )

    # normalize state
    arrays[1, 1] ./= S <: State ? sqrt(p) : p

    A(arrays)
end

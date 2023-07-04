using UUIDs: uuid4

"""
    ProjectedEntangledPair{P<:Plug,B<:Boundary} <: Quantum

A generic ansatz representing Projected Entangled Pair States (PEPS) and Projected Entangled Pair Operators (PEPO).
Type variable `P` represents the `Plug` type (`State` or `Operator`) and `B` represents the `Boundary` type (`Open` or `Periodic`).

# Ansatz Fields

  - `χ::Union{Nothing,Int}` Maximum virtual bond dimension.
"""
abstract type ProjectedEntangledPair{P,B} <: Quantum where {P<:Plug,B<:Boundary} end

boundary(::Type{<:ProjectedEntangledPair{P,B}}) where {P,B} = B
plug(::Type{<:ProjectedEntangledPair{P}}) where {P} = P

function ProjectedEntangledPair{P}(arrays; boundary::Type{<:Boundary} = Open, kwargs...) where {P<:Plug}
    ProjectedEntangledPair{P,boundary}(arrays; kwargs...)
end

metadata(T::Type{<:ProjectedEntangledPair}) = merge(metadata(supertype(T)), @NamedTuple begin
    χ::Union{Nothing,Int}
end)

function checkmeta(::Type{ProjectedEntangledPair{P,B}}, tn::TensorNetwork) where {P,B}
    # meta has correct value
    isnothing(tn.χ) || tn.χ > 0 || return false

    # no virtual index has dimensionality bigger than χ
    all(i -> isnothing(tn.χ) || size(tn, i) <= tn.χ, labels(tn, :virtual)) || return false

    return true
end

function _sitealias(::Type{ProjectedEntangledPair{P,Open}}, order, size, pos) where {P<:Plug}
    m, n = size
    i, j = pos

    order = [order...]

    filter(order) do dir
        !(i == 1 && dir === :u || i == m && dir === :d || j == 1 && dir === :l || j == n && dir === :r)
    end
end
_sitealias(::Type{ProjectedEntangledPair{P,Periodic}}, order, _, _) where {P<:Plug} = tuple(order...)

defaultorder(::Type{ProjectedEntangledPair{State}}) = (:l, :r, :u, :d, :o)
defaultorder(::Type{ProjectedEntangledPair{Operator}}) = (:l, :r, :u, :d, :i, :o)

"""
    ProjectedEntangledPair{P,B}(arrays::Matrix{AbstractArray}; χ::Union{Nothing,Int} = nothing, order = defaultorder(ProjectedEntangledPair{P}))

Construct a [`TensorNetwork`](@ref) with [`ProjectedEntangledPair`](@ref) ansatz, from the arrays of the tensors.

# Keyword Arguments

  - `χ` Maximum virtual bond dimension. Defaults to `nothing`.
  - `order` Order of the tensor indices on `arrays`. Defaults to `(:l, :r, :u, :d, :o)` if `P` is a `State`, `(:l, :r, :u, :d, :i, :o)` if `Operator`.
"""
function ProjectedEntangledPair{P,B}(
    arrays;
    χ = nothing,
    order = defaultorder(ProjectedEntangledPair{P}),
    metadata...,
) where {P<:Plug,B<:Boundary}
    issetequal(order, defaultorder(ProjectedEntangledPair{P})) || throw(
        ArgumentError(
            "`order` must be a permutation of $(join(String.(defaultorder(ProjectedEntangledPair{P})), ',', " and "))",
        ),
    )

    m, n = size(arrays)
    hinds = Dict((i, j) => Symbol(uuid4()) for i in 1:m, j in ringpeek(1:n))
    vinds = Dict((i, j) => Symbol(uuid4()) for i in ringpeek(1:m), j in 1:n)
    oinds = Dict((i, j) => Symbol(uuid4()) for i in 1:m, j in 1:n)
    iinds = Dict((i, j) => Symbol(uuid4()) for i in 1:m, j in 1:n)

    interlayer = if P <: State
        [Bijection(Dict(i + j * m => index for ((i, j), index) in oinds))]
    elseif P <: Operator
        [
            Bijection(Dict(i + j * m => index for ((i, j), index) in iinds)),
            Bijection(Dict(i + j * m => index for ((i, j), index) in oinds)),
        ]
    else
        throw(ErrorException("Plug $P is not valid"))
    end

    tensors = map(zip(Iterators.map(Tuple, eachindex(IndexCartesian(), arrays)), arrays)) do ((i, j), array)
        dirs = _sitealias(ProjectedEntangledPair{P,B}, order, (m, n), (i, j))

        labels = map(dirs) do dir
            if dir === :l
                hinds[(i, (mod1(j - 1, n), j))]
            elseif dir === :r
                hinds[(i, (j, mod1(j + 1, n)))]
            elseif dir === :u
                vinds[((mod1(i - 1, n), i), j)]
            elseif dir === :d
                vinds[((i, mod1(i + 1, n)), j)]
            elseif dir === :i
                iinds[(i, j)]
            elseif dir === :o
                oinds[(i, j)]
            end
        end
        alias = Dict(dir => label for (dir, label) in zip(dirs, labels))

        Tensor(array, labels; alias = alias)
    end |> vec

    return TensorNetwork{ProjectedEntangledPair{P,B}}(tensors; χ, plug = P, interlayer, metadata...)
end

const PEPS = ProjectedEntangledPair{State}
const PEPO = ProjectedEntangledPair{Operator}

# TODO normalize
# TODO let choose the orthogonality center
# TODO different input/output physical dims
function Base.rand(rng::Random.AbstractRNG, sampler::TNSampler{ProjectedEntangledPair{P,Open}}) where {P<:Plug}
    rows = sampler.rows
    cols = sampler.cols
    χ = sampler.χ
    p = get(sampler, :p, 2)
    T = get(sampler, :eltype, Float64)

    arrays::Matrix{AbstractArray{T,N} where {N}} = reshape(
        map(Iterators.product(1:rows, 1:cols)) do (i, j)
            shape = filter(
                !=(1),
                [
                    i === 1 ? 1 : χ,
                    i === rows ? 1 : χ,
                    j === 1 ? 1 : χ,
                    j === cols ? 1 : χ,
                    p,
                    if P <: State
                        1
                    elseif P <: Operator
                        p
                    else
                        throw(ErrorException("$P is not a valid Plug type"))
                    end,
                ],
            )

            rand(rng, T, shape...)
        end,
        rows,
        cols,
    )

    # normalize state
    arrays[1, 1] ./= P <: State ? sqrt(p) : p

    ProjectedEntangledPair{State,Open}(arrays; χ)
end

# TODO normalize
# TODO let choose the orthogonality center
# TODO different input/output physical dims
function Base.rand(rng::Random.AbstractRNG, sampler::TNSampler{ProjectedEntangledPair{P,Periodic}}) where {P<:Plug}
    rows = sampler.rows
    cols = sampler.cols
    χ = sampler.χ
    p = get(sampler, :p, 2)
    T = get(sampler, :eltype, Float64)

    arrays::Matrix{AbstractArray{T,N} where {N}} = reshape(
        map(Iterators.product(1:rows, 1:cols)) do (i, j)
            shape = tuple([χ, χ, χ, χ]..., ([if P <: State
                (p,)
            elseif P <: Operator
                (p, p)
            else
                throw(ErrorException("$P is not a valid Plug type"))
            end]...)...)

            # A = gramschmidt!(rand(rng, T, shape[1], prod(shape[1:end])))
            A = rand(rng, T, shape...)
        end,
        rows,
        cols,
    )

    # normalize state
    arrays[1, 1] ./= P <: State ? sqrt(p) : p

    ProjectedEntangledPair{State,Periodic}(arrays; χ)
end

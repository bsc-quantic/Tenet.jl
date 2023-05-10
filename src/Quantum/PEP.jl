using UUIDs: uuid4

abstract type ProjectedEntangledPair{P,B} <: Quantum where {P<:Plug,B<:Boundary} end

boundary(::Type{<:ProjectedEntangledPair{P,B}}) where {P,B} = B
plug(::Type{<:ProjectedEntangledPair{P}}) where {P} = P

function ProjectedEntangledPair{P}(arrays; boundary::Type{<:Boundary} = Open, kwargs...) where {P<:Plug}
    ProjectedEntangledPair{P,boundary}(arrays; kwargs...)
end

function checkmeta(::Type{ProjectedEntangledPair{P,B}}, tn::TensorNetwork) where {P,B}
    # TODO
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

function ProjectedEntangledPair{P,B}(
    arrays::Matrix;
    order = defaultorder(ProjectedEntangledPair{P}),
    metadata...,
) where {P<:Plug,B<:Boundary}
    issetequal(order, defaultorder(ProjectedEntangledPair{P})) || throw(
        ArgumentError(
            "`order` must be a permutation of $(join(String.(defaultorder(ProjectedEntangledPair{P})), ',', " and "))",
        ),
    )

    m, n = size(arrays)
    hinds = Dict((i, j) => Symbol(uuid4()) for i in 1:m for j in ringpeek(1:n))
    vinds = Dict((i, j) => Symbol(uuid4()) for i in ringpeek(1:m) for j in 1:n)
    oinds = Dict((i, j) => Symbol(uuid4()) for i in 1:m for j in 1:n)
    iinds = Dict((i, j) => Symbol(uuid4()) for i in 1:m for j in 1:n)

    # mark plug connectors
    plug = Dict((site, :out) => label for (site, label) in oinds)
    # TODO mark input plugs

    tensors = map(zip(Iterators.map(Tuple, eachindex(arrays)), arrays)) do ((i, j), array)
        dirs = _sitealias(ProjectedEntangledPair{P,B}, order, (m, n), (i, j))

        labels = map(dirs) do dir
            if dir === :l
                hinds[((mod1(i - 1, n), i), j)]
            elseif dir === :r
                hinds[((i, mod1(i + 1, n)), j)]
            elseif dir === :u
                vinds[(i, (mod1(j - 1, n), j))]
            elseif dir === :d
                vinds[(i, (j, mod1(j + 1, n)))]
            elseif dir === :i
                iinds[(i, j)]
            elseif dir === :o
                oinds[(i, j)]
            end
        end
        alias = Dict(dir => label for (dir, label) in zip(dirs, labels))

        Tensor(array, labels; alias = alias)
    end

    return TensorNetwork{ProjectedEntangledPair{P,B}}(tensors; plug, metadata...)
end
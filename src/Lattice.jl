using Graphs: Graphs
using BijectiveDicts: BijectiveDict

struct Bond{L<:AbstractLane}
    src::L
    dst::L
end

Base.convert(::Type{Bond}, edge::Pair{<:AbstractLane,<:AbstractLane}) = Bond(edge.first, edge.second)

Base.:(==)(a::Bond, b::Bond) = a.src == b.src && a.dst == b.dst || a.src == b.dst && a.dst == b.src

Graphs.src(edge::Bond) = edge.src
Graphs.dst(edge::Bond) = edge.dst
Graphs.reverse(edge::Bond) = Bond(Graphs.dst(edge), Graphs.src(edge))
Base.show(io::IO, edge::Bond) = write(io, "Bond: $(Graphs.src(edge)) - $(Graphs.dst(edge))")

Pair(e::Bond) = src(e) => dst(e)
Tuple(e::Bond) = (src(e), dst(e))

"""
    Lattice

A lattice is a graph where the vertices are [`Site`](@ref)s and the edges are virtual bonds.
It is used for representing the topology of a [`Ansatz`](@ref) Tensor Network.
It fulfills the [`AbstractGraph`](https://juliagraphs.org/Graphs.jl/stable/core_functions/interface/) interface.
"""
struct Lattice <: Graphs.AbstractGraph{Lane}
    mapping::BijectiveDict{Lane,Int,Dict{Lane,Int},Dict{Int,Lane}}
    graph::Graphs.SimpleGraph{Int} # TODO replace graph format because `rem_vertex!` renames vertices
end

Base.copy(lattice::Lattice) = Lattice(copy(lattice.mapping), copy(lattice.graph))
Base.:(==)(a::Lattice, b::Lattice) = a.mapping == b.mapping && a.graph == b.graph

# TODO these where needed by ChainRulesTestUtils, do we still need them?
Base.zero(::Type{Lattice}) = Lattice(BijectiveDict{Lane,Int}(), zero(Graphs.SimpleGraph{Int}))
Base.zero(::Lattice) = zero(Lattice)

Base.in(v::Lane, lattice::Lattice) = v ∈ keys(lattice.mapping)
Base.in(v::Site, lattice::Lattice) = Lane(v) ∈ keys(lattice.mapping)
Base.in(e::Bond, lattice::Lattice) = e ∈ edges(lattice)

Graphs.is_directed(::Type{Lattice}) = false

"""
    Graphs.vertices(::Lattice)

Return the vertices of the lattice; i.e. the list of [`Lane`](@ref)s.
"""
function Graphs.vertices(lattice::Lattice)
    return map(Graphs.vertices(lattice.graph)) do vertex
        lattice.mapping'[vertex]
    end
end

"""
    Graphs.edges(::Lattice)

Return the edges of the lattice; i.e. pairs of [`Lane`](@ref)s.
"""
Graphs.edges(lattice::Lattice) = BondIterator(Graphs.edges(lattice.graph), lattice)

"""
    Graphs.nv(::Lattice)

Return the number of vertices; i.e. [`Lane`](@ref)s, in the lattice.
"""
Graphs.nv(lattice::Lattice) = Graphs.nv(lattice.graph)

"""
    Graphs.ne(::Lattice)

Return the number of edges in the lattice.
"""
Graphs.ne(lattice::Lattice) = Graphs.ne(lattice.graph)

"""
    Graphs.has_vertex(lattice::Lattice, lane::AbstractLane)

Return `true` if the lattice has the given [`Lane`](@ref).
"""
Graphs.has_vertex(lattice::Lattice, lane::AbstractLane) = haskey(lattice.mapping, lane)

"""
    Graphs.has_edge(lattice::Lattice, edge)
    Graphs.has_edge(lattice::Lattice, a::Lane, b::Lane)

Return `true` if the lattice has the given edge.
"""
Graphs.has_edge(lattice::Lattice, edge::Bond) = Graphs.has_edge(lattice, edge.src, edge.dst)
function Graphs.has_edge(lattice::Lattice, a::AbstractLane, b::AbstractLane)
    return Graphs.has_vertex(lattice, a) &&
           Graphs.has_vertex(lattice, b) &&
           Graphs.has_edge(lattice.graph, lattice.mapping[a], lattice.mapping[b])
end

"""
    Graphs.neighbors(lattice::Lattice, lane::AbstractLane)

Return the neighbors [`Lane`](@ref)s of the given [`Lane`](@ref).
"""
function Graphs.neighbors(lattice::Lattice, lane::AbstractLane)
    Graphs.has_vertex(lattice, lane) || throw(ArgumentError("lane not in lattice"))
    vertex = lattice.mapping[lane]
    return map(Graphs.neighbors(lattice.graph, vertex)) do neighbor
        lattice.mapping'[neighbor]
    end
end

struct BondIterator <: Graphs.AbstractEdgeIter
    simpleit::Graphs.SimpleGraphs.SimpleEdgeIter{Graphs.SimpleGraph{Int}}
    lattice::Lattice
end

Graphs.ne(iterator::BondIterator) = Graphs.ne(iterator.lattice)
Base.eltype(::Type{BondIterator}) = Bond
Base.length(iterator::BondIterator) = length(iterator.simpleit)
Base.in(e::Bond, it::BondIterator) = Graphs.has_edge(it.lattice, Graphs.src(e), Graphs.src(dst))
Base.show(io::IO, iterator::BondIterator) = write(io, "BondIterator $(Graphs.ne(iterator))")

function Base.iterate(iterator::BondIterator, state=nothing)
    itres = isnothing(state) ? iterate(iterator.simpleit) : iterate(iterator.simpleit, state)
    isnothing(itres) && return nothing
    edge, state = itres
    return Bond(iterator.lattice.mapping'[Graphs.src(edge)], iterator.lattice.mapping'[Graphs.dst(edge)]), state
end

"""
    Lattice(::Val{:chain}, n; periodic=false)

Create a chain lattice with `n` sites.
"""
function Lattice(::Val{:chain}, n; periodic=false)
    graph = periodic ? Graphs.cycle_graph(n) : Graphs.path_graph(n)
    mapping = BijectiveDict{Lane,Int}([Lane(i) => i for i in 1:n])
    Lattice(mapping, graph)
end

"""
    Lattice(::Val{:rectangular}, nrows, ncols; periodic=false)

Create a rectangular lattice with `nrows` rows and `ncols` columns.
"""
function Lattice(::Val{:rectangular}, nrows, ncols; periodic=false)
    graph = Graphs.grid((nrows, ncols); periodic)
    mapping = BijectiveDict{Lane,Int}([Lane(row, col) => row + (col - 1) * nrows for row in 1:nrows for col in 1:ncols])
    Lattice(mapping, graph)
end

"""
    Lattice(::Val{:lieb}, nrows, ncols)

Create a Lieb lattice with `nrows` cell rows and `ncols` cell columns.
"""
function Lattice(::Val{:lieb}, ncellrows, ncellcols)
    nrows, ncols = 1 .+ 2 .* (ncellrows, ncellcols)

    lanes = [Lane(row, col) for row in 1:nrows for col in 1:ncols if !(row % 2 == 0 && col % 2 == 0)]
    mapping = BijectiveDict{Lane,Int}([lane => i for (i, lane) in enumerate(lanes)])
    graph = Graphs.SimpleGraph{Int}(length(lanes))

    # add horizontal edges
    for row in 1:2:nrows, col in 1:(ncols - 1)
        i = mapping[Lane(row, col)]
        j = mapping[Lane(row, col + 1)]
        Graphs.add_edge!(graph, i, j)
    end

    # add vertical edges
    for row in 1:(nrows - 1), col in 1:2:ncols
        i = mapping[Lane(row, col)]
        j = mapping[Lane(row + 1, col)]
        Graphs.add_edge!(graph, i, j)
    end

    return Lattice(mapping, graph)
end

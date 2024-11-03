using Graphs
using BijectiveDicts: BijectiveIdDict

struct LatticeEdge <: AbstractEdge{Site}
    src::Site
    dst::Site
end

Base.convert(::Type{LatticeEdge}, edge::Pair{<:Site,<:Site}) = LatticeEdge(edge.first, edge.second)

Graphs.src(edge::LatticeEdge) = edge.src
Graphs.dst(edge::LatticeEdge) = edge.dst
Graphs.reverse(edge::LatticeEdge) = LatticeEdge(dst(edge), src(edge))
Base.show(io::IO, edge::LatticeEdge) = write(io, "LatticeEdge $(src(edge)) → $(dst(edge))")

Pair(e::LatticeEdge) = src(e) => dst(e)
Tuple(e::LatticeEdge) = (src(e), dst(e))

"""
    Lattice

A lattice is a graph where the vertices are [`Site`](@ref)s and the edges are virtual bonds.
"""
struct Lattice <: AbstractGraph{Site}
    mapping::BijectiveIdDict{Site,Int}
    graph::Graphs.SimpleGraph{Int}
end

Base.copy(lattice::Lattice) = Lattice(copy(lattice.mapping), copy(lattice.graph))
Base.:(==)(a::Lattice, b::Lattice) = a.mapping == b.mapping && a.graph == b.graph

Base.zero(::Type{Lattice}) = Lattice(BijectiveIdDict{Site,Int}(), zero(Graphs.SimpleGraph{Int}))
Base.zero(::Lattice) = zero(Lattice)
Graphs.is_directed(::Type{Lattice}) = false

function Graphs.vertices(lattice::Lattice)
    return map(vertices(lattice.graph)) do vertex
        lattice.mapping'[vertex]
    end
end

Graphs.edges(lattice::Lattice) = LatticeEdgeIterator(edges(lattice.graph), lattice)

Graphs.nv(lattice::Lattice) = nv(lattice.graph)
Graphs.ne(lattice::Lattice) = ne(lattice.graph)

Graphs.has_vertex(lattice::Lattice, site::Site) = haskey(lattice.mapping, site)
Graphs.has_edge(lattice::Lattice, edge::LatticeEdge) = has_edge(lattice, edge.src, edge.dst)
function Graphs.has_edge(lattice::Lattice, a::Site, b::Site)
    return has_vertex(lattice, a) &&
           has_vertex(lattice, b) &&
           has_edge(lattice.graph, lattice.mapping[a], lattice.mapping[b])
end

function Graphs.neighbors(lattice::Lattice, site::Site)
    has_vertex(lattice, site) || throw(ArgumentError("site not in lattice"))
    vertex = lattice.mapping[site]
    return map(neighbors(lattice.graph, vertex)) do neighbor
        lattice.mapping'[neighbor]
    end
end

struct LatticeEdgeIterator <: Graphs.AbstractEdgeIter
    simpleit::Graphs.SimpleGraphs.SimpleEdgeIter{Graphs.SimpleGraph{Int}}
    lattice::Lattice
end

Graphs.ne(iterator::LatticeEdgeIterator) = ne(iterator.lattice)
Base.eltype(::Type{LatticeEdgeIterator}) = LatticeEdge
Base.length(iterator::LatticeEdgeIterator) = length(iterator.simpleit)
Base.in(e::LatticeEdge, it::LatticeEdgeIterator) = has_edge(it.lattice, src(e), src(dst))
Base.show(io::IO, iterator::LatticeEdgeIterator) = write(io, "LatticeEdgeIterator $(ne(iterator))")

function Base.iterate(iterator::LatticeEdgeIterator, state=nothing)
    itres = isnothing(state) ? iterate(iterator.simpleit) : iterate(iterator.simpleit, state)
    isnothing(itres) && return nothing
    edge, state = itres
    return LatticeEdge(iterator.lattice.mapping'[src(edge)], iterator.lattice.mapping'[dst(edge)]), state
end

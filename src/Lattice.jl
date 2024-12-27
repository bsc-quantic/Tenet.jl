using Graphs: Graphs
using BijectiveDicts: BijectiveIdDict

struct LatticeEdge <: Graphs.AbstractEdge{Site}
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
It is used for representing the topology of a [`Ansatz`](@ref) Tensor Network.
It fulfills the [`AbstractGraph`](https://juliagraphs.org/Graphs.jl/stable/core_functions/interface/) interface.
"""
struct Lattice <: Graphs.AbstractGraph{Site}
    mapping::BijectiveIdDict{Site,Int}
    graph::Graphs.SimpleGraph{Int}
end

Base.copy(lattice::Lattice) = Lattice(copy(lattice.mapping), copy(lattice.graph))
Base.:(==)(a::Lattice, b::Lattice) = a.mapping == b.mapping && a.graph == b.graph

# TODO these where needed by ChainRulesTestUtils, do we still need them?
Base.zero(::Type{Lattice}) = Lattice(BijectiveIdDict{Site,Int}(), zero(Graphs.SimpleGraph{Int}))
Base.zero(::Lattice) = zero(Lattice)

Graphs.is_directed(::Type{Lattice}) = false

"""
    Graphs.vertices(::Lattice)

Return the vertices of the lattice; i.e. the list of [`Site`](@ref)s.
"""
function Graphs.vertices(lattice::Lattice)
    return map(Graphs.vertices(lattice.graph)) do vertex
        lattice.mapping'[vertex]
    end
end

"""
    Graphs.edges(::Lattice)

Return the edges of the lattice; i.e. pairs of [`Site`](@ref)s.
"""
Graphs.edges(lattice::Lattice) = LatticeEdgeIterator(Graphs.edges(lattice.graph), lattice)

"""
    Graphs.nv(::Lattice)

Return the number of vertices/[`Site`](@ref)s in the lattice.
"""
Graphs.nv(lattice::Lattice) = Graphs.nv(lattice.graph)

"""
    Graphs.ne(::Lattice)

Return the number of edges in the lattice.
"""
Graphs.ne(lattice::Lattice) = Graphs.ne(lattice.graph)

"""
    Graphs.has_vertex(lattice::Lattice, site::Site)

Return `true` if the lattice has the given [`Site`](@ref).
"""
Graphs.has_vertex(lattice::Lattice, site::Site) = haskey(lattice.mapping, site)

"""
    Graphs.has_edge(lattice::Lattice, edge)
    Graphs.has_edge(lattice::Lattice, a::Site, b::Site)

Return `true` if the lattice has the given edge.
"""
Graphs.has_edge(lattice::Lattice, edge::LatticeEdge) = Graphs.has_edge(lattice, edge.src, edge.dst)
function Graphs.has_edge(lattice::Lattice, a::Site, b::Site)
    return Graphs.has_vertex(lattice, a) &&
           Graphs.has_vertex(lattice, b) &&
           Graphs.has_edge(lattice.graph, lattice.mapping[a], lattice.mapping[b])
end

"""
    Graphs.neighbors(lattice::Lattice, site::Site)

Return the neighbors [`Site`](@ref)s of the given [`Site`](@ref).
"""
function Graphs.neighbors(lattice::Lattice, site::Site)
    Graphs.has_vertex(lattice, site) || throw(ArgumentError("site not in lattice"))
    vertex = lattice.mapping[site]
    return map(Graphs.neighbors(lattice.graph, vertex)) do neighbor
        lattice.mapping'[neighbor]
    end
end

struct LatticeEdgeIterator <: Graphs.AbstractEdgeIter
    simpleit::Graphs.SimpleGraphs.SimpleEdgeIter{Graphs.SimpleGraph{Int}}
    lattice::Lattice
end

Graphs.ne(iterator::LatticeEdgeIterator) = Graphs.ne(iterator.lattice)
Base.eltype(::Type{LatticeEdgeIterator}) = LatticeEdge
Base.length(iterator::LatticeEdgeIterator) = length(iterator.simpleit)
Base.in(e::LatticeEdge, it::LatticeEdgeIterator) = Graphs.has_edge(it.lattice, Graphs.src(e), Graphs.src(dst))
Base.show(io::IO, iterator::LatticeEdgeIterator) = write(io, "LatticeEdgeIterator $(ne(iterator))")

function Base.iterate(iterator::LatticeEdgeIterator, state=nothing)
    itres = isnothing(state) ? iterate(iterator.simpleit) : iterate(iterator.simpleit, state)
    isnothing(itres) && return nothing
    edge, state = itres
    return LatticeEdge(iterator.lattice.mapping'[Graphs.src(edge)], iterator.lattice.mapping'[Graphs.dst(edge)]), state
end

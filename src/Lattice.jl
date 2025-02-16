using Graphs: Graphs

struct Bond{L<:AbstractLane}
    src::L
    dst::L

    Bond(src::L, dst::L) where {L<:AbstractLane} = new{L}(minmax(src, dst)...)
end

Base.convert(::Type{Bond}, edge::Pair{<:AbstractLane,<:AbstractLane}) = Bond(edge.first, edge.second)

Base.:(==)(a::Bond, b::Bond) = a.src == b.src && a.dst == b.dst || a.src == b.dst && a.dst == b.src

Graphs.src(edge::Bond) = edge.src
Graphs.dst(edge::Bond) = edge.dst
Graphs.reverse(edge::Bond) = Bond(Graphs.dst(edge), Graphs.src(edge))
Base.show(io::IO, edge::Bond) = write(io, "Bond: $(Graphs.src(edge)) - $(Graphs.dst(edge))")

Pair(e::Bond) = Graphs.src(e) => Graphs.dst(e)
Tuple(e::Bond) = (Graphs.src(e), Graphs.dst(e))

function Base.iterate(bond::Bond, state=0)
    if state == 0
        (Graphs.src(bond), 1)
    elseif state == 1
        (Graphs.dst(bond), 2)
    else
        nothing
    end
end

Base.IteratorSize(::Type{Bond}) = Base.HasLength()
Base.length(::Bond) = 2
Base.IteratorEltype(::Type{Bond{L}}) where {L} = Base.HasEltype()
Base.eltype(::Bond{L}) where {L} = L
Base.isdone(::Bond, state) = state == 2

"""
    Lattice

A lattice is a graph where the vertices are [`Site`](@ref)s and the edges are virtual bonds.
It is used for representing the topology of a [`Ansatz`](@ref) Tensor Network.
It fulfills the [`AbstractGraph`](https://juliagraphs.org/Graphs.jl/stable/core_functions/interface/) interface.
"""
struct Lattice <: Graphs.AbstractGraph{Lane}
    lanes::Vector{Lane}
    graph::Graphs.SimpleGraph{Int}

    function Lattice(lanes, graph)
        length(lanes) == Graphs.nv(graph) || throw(ArgumentError("number of lanes must match number of vertices"))
        new(lanes, graph)
    end
end

Lattice() = Lattice(Lane[], Graphs.SimpleGraph{Int}())
Lattice(lanes) = Lattice(lanes, Graphs.SimpleGraph{Int}(length(lanes)))

Base.parent(lattice::Lattice) = lattice.graph
parent_vertex(lattice::Lattice, lane::Lane) = findfirst(==(lane), lattice.lanes)
child_vertex(lattice::Lattice, vertex::Int) = lattice.lanes[vertex]

Base.copy(lattice::Lattice) = Lattice(copy(lattice.lanes), copy(lattice.graph))
Base.:(==)(a::Lattice, b::Lattice) = a.lanes == b.lanes && a.graph == b.graph

# TODO these where needed by ChainRulesTestUtils, do we still need them?
Base.zero(::Type{Lattice}) = Lattice(Lane[], zero(Graphs.SimpleGraph{Int}))
Base.zero(::Lattice) = zero(Lattice)

Base.in(v::Lane, lattice::Lattice) = v ∈ lattice.lanes
Base.in(v::Site, lattice::Lattice) = Lane(v) ∈ lattice
Base.in(e::Bond, lattice::Lattice) = e ∈ edges(lattice)

Graphs.is_directed(::Type{Lattice}) = false

function Graphs.add_vertex!(lattice::Lattice, lane::Lane)
    if !Graphs.add_vertex!(parent(lattice))
        throw(ErrorException("Could not add vertex to parent graph"))
    end
    push!(lattice.lanes, lane)
    return lattice
end

function Graphs.add_edge!(lattice::Lattice, a::Lane, b::Lane)
    if !Graphs.add_edge!(lattice.graph, parent_vertex(lattice, a), parent_vertex(lattice, b))
        throw(ErrorException("Could not add edge to parent graph"))
    end
    return lattice
end

function Graphs.rem_vertex!(lattice::Lattice, lane::Lane)
    vertex = parent_vertex(lattice, lane)
    res = Graphs.remove_vertex!(lattice.graph, vertex)
    res && deleteat!(lattice.lanes, vertex)
    return res
end

Graphs.rem_edge!(lattice::Lattice, edge::Bond) = Graphs.rem_edge!(lattice, Graphs.src(edge), Graphs.dst(edge))
function Graphs.rem_edge!(lattice::Lattice, a::Lane, b::Lane)
    code_a = parent_vertex(lattice, a)
    code_b = parent_vertex(lattice, b)

    Graphs.rem_edge!(lattice.graph, Graphs.SimpleEdge(code_a, code_b))
end

"""
    Graphs.vertices(::Lattice)

Return the vertices of the lattice; i.e. the list of [`Lane`](@ref)s.
"""
Graphs.vertices(lattice::Lattice) = Tuple(lattice.lanes)

"""
    Graphs.edges(::Lattice)

Return the edges of the lattice; i.e. pairs of [`Lane`](@ref)s.
"""
Graphs.edges(lattice::Lattice) = BondIterator(Graphs.edges(parent(lattice)), lattice)

"""
    Graphs.nv(::Lattice)

Return the number of vertices; i.e. [`Lane`](@ref)s, in the lattice.
"""
Graphs.nv(lattice::Lattice) = Graphs.nv(parent(lattice))

"""
    Graphs.ne(::Lattice)

Return the number of edges in the lattice.
"""
Graphs.ne(lattice::Lattice) = Graphs.ne(parent(lattice))

"""
    Graphs.has_vertex(lattice::Lattice, lane::AbstractLane)

Return `true` if the lattice has the given [`Lane`](@ref).
"""
Graphs.has_vertex(lattice::Lattice, lane::AbstractLane) = lane ∈ lattice

"""
    Graphs.has_edge(lattice::Lattice, edge)
    Graphs.has_edge(lattice::Lattice, a::Lane, b::Lane)

Return `true` if the lattice has the given edge.
"""
Graphs.has_edge(lattice::Lattice, edge::Bond) = Graphs.has_edge(lattice, edge.src, edge.dst)
function Graphs.has_edge(lattice::Lattice, a::AbstractLane, b::AbstractLane)
    return Graphs.has_vertex(lattice, a) &&
           Graphs.has_vertex(lattice, b) &&
           Graphs.has_edge(parent(lattice), parent_vertex(lattice, a), parent_vertex(lattice, b))
end

"""
    Graphs.neighbors(lattice::Lattice, lane::AbstractLane)

Return the neighbors [`Lane`](@ref)s of the given [`Lane`](@ref).
"""
function Graphs.neighbors(lattice::Lattice, lane::AbstractLane)
    Graphs.has_vertex(lattice, lane) || throw(ArgumentError("lane not in lattice"))
    vertex = parent_vertex(lattice, lane)
    return map(Graphs.neighbors(parent(lattice), vertex)) do neighbor
        child_vertex(lattice, neighbor)
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
    a = child_vertex(iterator.lattice, Graphs.src(edge))
    b = child_vertex(iterator.lattice, Graphs.dst(edge))
    return Bond(a, b), state
end

"""
    Lattice(::Val{:chain}, n; periodic=false)

Create a chain lattice with `n` sites.
"""
function Lattice(::Val{:chain}, n; periodic=false)
    graph = periodic ? Graphs.cycle_graph(n) : Graphs.path_graph(n)
    lanes = [Lane(i) for i in 1:n]
    Lattice(lanes, graph)
end

"""
    Lattice(::Val{:rectangular}, nrows, ncols; periodic=false)

Create a rectangular lattice with `nrows` rows and `ncols` columns.
"""
function Lattice(::Val{:rectangular}, nrows, ncols; periodic=false)
    graph = Graphs.grid((nrows, ncols); periodic)
    lanes = vec([Lane(row, col) for col in 1:ncols for row in 1:nrows])
    Lattice(lanes, graph)
end

"""
    Lattice(::Val{:lieb}, nrows, ncols)

Create a Lieb lattice with `nrows` cell rows and `ncols` cell columns.
"""
function Lattice(::Val{:lieb}, ncellrows, ncellcols)
    lattice = Lattice()
    nrows, ncols = 1 .+ 2 .* (ncellrows, ncellcols)

    # add vertices
    for row in 1:nrows, col in 1:ncols
        # skip holes
        row % 2 == 0 && col % 2 == 0 && continue

        lane = Lane(row, col)
        Graphs.add_vertex!(lattice, lane)
    end

    # add horizontal edges
    for row in 1:2:nrows, col in 1:(ncols - 1)
        Graphs.add_edge!(lattice, Lane(row, col), Lane(row, col + 1))
    end

    # add vertical edges
    for row in 1:(nrows - 1), col in 1:2:ncols
        Graphs.add_edge!(lattice, Lane(row, col), Lane(row + 1, col))
    end

    return lattice
end

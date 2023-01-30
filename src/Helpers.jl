using Base.Iterators: Cycle, cycle

const Sequence{T} = Union{AbstractArray{T,1},NTuple{N,T} where {N}}

mutable struct RingPeek{Itr<:Cycle}
    const it::Itr
    base::Any
end

ringpeek(itr) = RingPeek(cycle(itr), nothing)
ringpeek(itr::Itr) where {Itr<:Cycle} = RingPeek(itr, nothing)

period(itr::Cycle) = length(itr.xs)
Base.IteratorSize(::Type{RingPeek{Itr}}) where {Itr} = Base.HasLength()
Base.length(itr::RingPeek{Itr}) where {Itr} = period(itr.it)

Base.IteratorEltype(::Type{RingPeek{Itr}}) where {Itr} = Base.IteratorEltype(Itr)
Base.eltype(::Type{RingPeek{Itr}}) where {Itr} = Tuple{eltype(Itr),eltype(Itr)}

Base.isdone(it::RingPeek, state) = it.base == state

function Base.iterate(it::RingPeek{Itr}) where {Itr}
    x, state = iterate(it.it)
    peeked, nextstate = iterate(it.it, state)
    it.base = state
    ((x, peeked), state)
end

function Base.iterate(it::RingPeek{Itr}, state) where {Itr}
    x, newstate = iterate(it.it, state)
    peeked, _ = iterate(it.it, newstate)

    newstate == it.base && return nothing

    ((x, peeked), newstate)
end

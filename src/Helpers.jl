using Base.Iterators: Cycle, cycle

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

const NUM_UNICODE_LETTERS = VERSION >= v"1.9" ? 136104 : 131756

"""
    letter(i)

Return `i`-th printable Unicode letter.

# Examples

```jldoctest; setup = :(letter = Tenet.letter)
julia> letter(1)
:A

julia> letter(27)
:a

julia> letter(249)
:ƃ

julia> letter(20204)
:櫛
```
"""
letter(i) =
    Iterators.drop(Iterators.filter(isletter, Iterators.map(Char, 1:2^21-1)), i - 1) |> iterate |> first |> Symbol

# NOTE from https://stackoverflow.com/q/54652787
function nonunique(x)
    uniqueindexes = indexin(unique(x), x)
    nonuniqueindexes = setdiff(1:length(x), uniqueindexes)
    unique(x[nonuniqueindexes])
end

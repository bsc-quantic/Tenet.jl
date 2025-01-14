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
letter(i) = Symbol(first(iterate(Iterators.drop(Iterators.filter(isletter, Iterators.map(Char, 1:(2^21 - 1))), i - 1))))

# NOTE from https://stackoverflow.com/q/54652787
function nonunique(x)
    uniqueindexes = indexin(unique(x), collect(x))
    nonuniqueindexes = setdiff(1:length(x), uniqueindexes)
    return Tuple(unique(x[nonuniqueindexes]))
end

struct IndexCounter
    counter::Threads.Atomic{Int}

    IndexCounter(init::Int=1) = new(Threads.Atomic{Int}(init))
end

currindex(gen::IndexCounter) = letter(gen.counter[])
function nextindex!(gen::IndexCounter)
    if gen.counter.value >= 135000
        throw(ErrorException("run-out of indices!"))
    end
    return letter(Threads.atomic_add!(gen.counter, 1))
end
resetinds!(gen::IndexCounter) = letter(Threads.atomic_xchg!(gen.counter, 1))

# eps wrapper so it handles Complex numbers
# if is Complex, extract the parametric type and get the eps of that
wrap_eps(x) = eps(x)
wrap_eps(::Type{Complex{T}}) where {T} = eps(T)

struct UnsafeScope
    refs::Vector{WeakRef}

    UnsafeScope() = new(Vector{WeakRef}())
end

Base.values(uc::UnsafeScope) = map(x -> x.value, uc.refs)

# from https://discourse.julialang.org/t/sort-keys-of-namedtuple/94630/3
@generated sort_nt(nt::NamedTuple{KS}) where {KS} = :(NamedTuple{$(Tuple(sort(collect(KS))))}(nt))

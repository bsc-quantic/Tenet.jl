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
    uniqueindexes = indexin(unique(x), x)
    nonuniqueindexes = setdiff(1:length(x), uniqueindexes)
    return unique(x[nonuniqueindexes])
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
resetindex!(gen::IndexCounter) = letter(Threads.atomic_xchg!(gen.counter, 1))

# eps wrapper so it handles Complex numbers
# if is Complex, extract the parametric type and get the eps of that
wrap_eps(x) = eps(x)
wrap_eps(::Type{Complex{T}}) = eps(T)

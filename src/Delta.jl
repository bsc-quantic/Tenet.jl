import Base: eltype, size, getindex

struct Delta{T}
    values::Vector{T}
    size::Vector{Int}
end

eltype(::Delta{T}) where {T} = T

size(a::Delta) = Tuple(a.size)

getindex(a::Delta{T}, elements::Int...) where {T} = reduce(==, elements) ? a.values[elements[1]] : T(0) # TODO delta 0
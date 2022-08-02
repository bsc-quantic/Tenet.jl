import Base: eltype, size, getindex, ndims, firstindex, lastindex

# TODO use StaticArray{T,N}?
struct Delta{T,N} <: AbstractArray{T,N}
    values::Vector{T}
end

δ = Delta

eltype(::Delta{T}) where {T} = T
ndims(A::Delta{T,N}) where {T,N} = N
size(A::Delta{T,N}) where {T,N} = ntuple(x -> length(A.values), N)

firstindex(A::Delta{T,N}) where {T,N} = ntuple(x -> 1, N)
lastindex(A::Delta{T,N}) where {T,N} = ntuple(x -> length(A.values), N)
getindex(A::Delta{T}, elements::Int...) where {T} =
    reduce(==, elements) ? A.values[first(elements)] : T(0) # TODO delta 0
setindex!(A::Delta{T}, X, ind) where {T} = begin
    A.values[ind] = X
end
setindex!(A::Delta{T}, X, inds...) where {T} = error("not implemented yet")

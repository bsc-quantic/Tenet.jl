import Base: eltype, size, stride, strides, ndims, axes, length, keys, conj, conj!
import Base: convert, show, kron
using Lazy

struct Tensor{T,N}
    data::AbstractArray{T,N}
    labels::Vector{Char}
    tags::Set{String}
end

Tensor(data; labels=rand(Char, N), tags=Set{String}()) where {T,N} = Tensor{T,N}(data, labels, tags)

Base.show(io::IO, t::Tensor) = print(io, "Tensor{$(eltype(t))}(data=$(t.data), labels=$(t.labels), tags=$(t.tags))")

@forward Tensor.data eltype, size, stride, strides, ndims, axes, length, keys, conj, conj!

permutedims(t::Tensor, perm) = Tensor(permutedims(t.array, perm), t.labels[perm])
permutedims(t::Tensor, perm::Vector{Char}) = begin
    p = map(c -> findfirst(x -> x == c, t.labels), perm)
    permutedims(t, p)
end

inds(x::Tensor) = x.labels
innerinds(a::Tensor, b::Tensor) = inds(a) ∩ inds(b)
outerinds(a::Tensor, b::Tensor) = setdiff(inds(a) ∪ inds(b), inds(a) ∩ inds(b))
reindex!(t::Tensor, mapping::Dict{Char,Char}) = error("not implemented yet")

# tags
tags(t::Tensor) = t.tags
addtag!(t::Tensor, tag::String) = push!(t.tags, tag)
rmtag!(t::Tensor, tag::String) = pop!(t.tags, tag)
droptags!(t::Tensor) = empty!(t.tags)

# conversion
convert(::Type{Array}, t::Tensor) = t.array
promote_rule(::Type{Tensor}, ::Type{Array}) = Array

# kronecker product
kron(A::Array{T,N}, B::Array{T,M}) where {T,N,M} = reshape(kron(vec(A), vec(B)), size(A)..., size(B)...)
kron(A::Tensor{T}, B::Tensor{T}) where {T} = Tensor(kron(A.data, B.data), inds(A) ∪ inds(B), tags(A) ∪ tags(B))

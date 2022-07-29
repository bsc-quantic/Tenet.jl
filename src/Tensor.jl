import Base: eltype, size, stride, strides, ndims, axes, length, keys, conj, conj!
import Base: convert, show, kron
using Lazy

struct Tensor{T}
    data::Array{T}
    labels::Vector{Char}
    tags::Set{String}

    # TODO do not use `rand` for `labels`
    Tensor(data::Array{T}, labels=nothing, tags=Set{String}()) where {T} = new{T}(
        data,
        labels == nothing ? rand(Char, ndims(data)) : labels,
        tags)
end

Base.show(io::IO, t::Tensor) = print(io, "Tensor{$(eltype(t))}(data=$(t.data), labels=$(t.labels), tags=$(t.tags))")

@forward Tensor.data eltype, size, stride, strides, ndims, axes, length, keys, conj, conj!

permutedims(t::Tensor, perm) = Tensor(permutedims(t.array, perm), t.labels[perm])
permutedims(t::Tensor, perm::Vector{Char}) = begin
    p = map(c -> findfirst(x -> x == c, t.labels), perm)
    permutedims(t, p)
end

innerinds(a::Tensor, b::Tensor) = a.labels ∩ b.labels
outerinds(a::Tensor, b::Tensor) = setdiff(a.labels ∪ b.labels, a.labels ∩ b.labels)
reindex!(t::Tensor, mapping::Dict{Char,Char}) = error("not implemented yet")

# tags
droptags!(t::Tensor) = empty!(t.tags)

# conversion
convert(::Type{Array}, t::Tensor) = t.array
promote_rule(::Type{Tensor}, ::Type{Array}) = Array

# kronecker product
kron(A::Array{T,N}, B::Array{T,M}) where {T,N,M} = reshape(kron(vec(A), vec(B))size(A)..., size(B)...)
kron(A::Tensor{T}, B::Tensor{T}) where {T} = Tensor(reshape(kron(A.data, B.data), size(A)..., size(B)...), A.labels)

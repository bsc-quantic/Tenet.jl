import Base: convert
using ReusePatterns
using TensorOperations

struct Tensor{T}
    array::Array{T}
    labels::Vector{Char}
end

@forward((Tensor, :array), AbstractArray)

permutedims(t::Tensor, perm) = Tensor(permutedims(t.array, perm), t.labels[perm])
permutedims(t::Tensor, perm::Vector{Char}) = begin
    p = map(c -> findfirst(x -> x == c, t.labels), perm)
    permutedims(t, p)
end

inner_inds(a::Tensor, b::Tensor) = a.labels ∩ b.labels
outer_inds(a::Tensor, b::Tensor) = setdiff(a.labels ∪ b.labels, a.labels ∩ b.labels)

# contract
# @traits contract(a::Tensor, b::Tensor) where {isempty(inner_inds(a, b))} = begin
# end

# conversion
convert(::Type{Array}, t::Tensor) = t.array
promote_rule(::Type{Tensor}, ::Type{Array}) = Array
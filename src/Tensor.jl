import Base: convert, show
using ReusePatterns

struct Tensor{T}
    data::Array{T}
    labels::Vector{Char}
    tags::Set{String}
end

Base.show(io::IO, t::Tensor) = print(io, "Tensor{$(eltype(t))}(data=$t.data, labels=$t.labels, tags=$t.tags)")

@forward((Tensor, :data), AbstractArray)

permutedims(t::Tensor, perm) = Tensor(permutedims(t.array, perm), t.labels[perm])
permutedims(t::Tensor, perm::Vector{Char}) = begin
    p = map(c -> findfirst(x -> x == c, t.labels), perm)
    permutedims(t, p)
end

inner_inds(a::Tensor, b::Tensor) = a.labels ∩ b.labels
outer_inds(a::Tensor, b::Tensor) = setdiff(a.labels ∪ b.labels, a.labels ∩ b.labels)
reindex!(t::Tensor, mapping::Dict{Char,Char}) = error("not implemented yet")

# tags
droptags!(t::Tensor) = empty!(t.tags)

# conversion
convert(::Type{Array}, t::Tensor) = t.array
promote_rule(::Type{Tensor}, ::Type{Array}) = Array
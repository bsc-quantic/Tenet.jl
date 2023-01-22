struct Index
    name::Symbol
    size::Int
    links::Vector{Tensor}
    meta::Dict{Symbol,Any}

    function Index(name, size; meta...)
        size < 1 && throw(DomainError(size, "size must be >1"))

        links = Tensor[]

        meta = Dict{Symbol,Any}(meta)
        !haskey(meta, :site) && (meta[:site] = nothing)
        !haskey(meta, :tags) && (meta[:tags] = Set{String}())

        return new(name, size, links, meta)
    end
end

Base.copy(i::Index) = Index(nameof(i), size(i); i.meta...)

Base.nameof(i::Index) = i.name
Base.size(i::Index) = i.size

function checksize(i::Index)
    checklinks(i) && all(t -> size(t, findfirst(j -> i === j, indices(t))) == i.size, i.links)
end

Base.show(io::IO, i::Index) = show(io, i.name)

# NOTE extends `dim` on `Tensor` after `Index` definition
dim(t::Tensor, i::Index) = dim(t, nameof(i))

site(i::Index) = i.meta[:site]
isphysical(i::Index) = !isnothing(site(i))
isvirtual(i::Index) = !isphysical(i)

tags(i::Index) = i.meta[:tags]
tag!(i::Index, tag::String) = push!(tags(i), tag)
hastag(i::Index, tag::String) = tag ∈ tags(i)
untag!(i::Index, tag::String) = delete!(tags(i), tag)

# TODO test these!
# NOTE return copy so on info is not lost while modification (e.g. HyperindsConverter transformation)
links(i::Index) = i.links |> copy
function link!(i::Index, t)
    size(i) != size(t, i) && throw(
        DimensionMismatch("size of index $i ($(size(i))) is not equal to size of index $i at tensor ($(size(t,i)))"),
    )
    push!(i.links, t)
end
unlink!(i::Index, t) = filter!(x -> x != t, i.links)

checklinks(i::Index) = all((∋(i) ∘ inds), links(i))

isopenind(i::Index) = length(links(i)) == 1
ishyperind(i::Index) = length(links(i)) > 2

Base.:(==)(i::Index, j::Index) = size(i) == size(j) && ((site(i) == site(j) !== nothing) || (nameof(size) == nameof(j)))

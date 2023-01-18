struct Index
    name::Symbol
    size::Int
    links::Vector{Any}
    meta::Dict{Symbol,Any}

    function Index(name, size; meta...)
        size < 1 && throw(BoundsError("size must be >=1 ($size < 1)"))

        links = Any[]

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

site(i::Index) = i.meta[:site]
isphysical(i::Index) = !isnothing(site(i))
isvirtual(i::Index) = !isphysical(i)

tags(i::Index) = i.meta[:tags]
tag!(i::Index, tag::String) = push!(tags(i), tag)
hastag(i::Index, tag::String) = tag ∈ tags(i)
untag!(i::Index, tag::String) = delete!(tags(i), tag)

# TODO test these!
links(i::Index) = i.links
function link!(i::Index, t)
    size(i) != size(t, i) && throw(
        DimensionMismatch("size of index $i ($(size(i))) is not equal to size of index $i at tensor ($(size(t,i)))"),
    )
    push!(links(i), t)
end
unlink!(i::Index, t) = filter!(x -> x != t, links(i))

checklinks(i::Index) = all((∋(i) ∘ inds), links(i))

isopenindex(i::Index) = length(links(i)) == 1
ishyperindex(i::Index) = length(links(i)) > 2

Base.:(==)(i::Index, j::Index) = (site(i) == site(j) !== nothing) || a.name == b.name

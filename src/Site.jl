# TODO Should we store here some information about quantum numbers?
"""
    Site(id[, dual = false])
    site"i,j,..."

Represents a physical index.
"""
struct Site{N}
    id::NTuple{N,Int}
    dual::Bool

    Site(id::NTuple{N,Int}; dual=false) where {N} = new{N}(id, dual)
end

Site(id::Int; kwargs...) = Site((id,); kwargs...)
Site(id::Vararg{Int,N}; kwargs...) where {N} = Site(id; kwargs...)

id(site::Site{1}) = only(site.id)
id(site::Site) = site.id

Base.CartesianIndex(site::Site) = CartesianIndex(id(site))

isdual(site::Site) = site.dual
Base.show(io::IO, site::Site) = print(io, "$(id(site))$(site.dual ? "'" : "")")
Base.adjoint(site::Site) = Site(id(site); dual=!site.dual)
Base.isless(a::Site, b::Site) = id(a) < id(b)

macro site_str(str)
    m = match(r"^(\d+,)*\d+('?)$", str)
    if isnothing(m)
        error("Invalid site string: $str")
    end

    id = tuple(map(eachmatch(r"(\d+)", str)) do match
        parse(Int, only(match.captures))
    end...)

    dual = endswith(str, "'")

    return :(Site($id; dual=$dual))
end

Base.zero(x::Dict{Site,Symbol}) = x

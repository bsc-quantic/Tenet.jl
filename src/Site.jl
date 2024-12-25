"""
    Site(id[; dual = false])
    Site(i1, i2, ...[; dual = false])
    site"i,j,...[']"

Represents the location of a physical index. `Site` objects are used to label the indices of tensors in a [`Quantum`](@ref) tensor network.
They are

See also: [`sites`](@ref), [`id`](@ref), [`isdual`](@ref)
"""
struct Site{N}
    id::NTuple{N,Int}
    dual::Bool

    Site(id::NTuple{N,Int}; dual=false) where {N} = new{N}(id, dual)
end

Site(id::Int; kwargs...) = Site((id,); kwargs...)
Site(id::Vararg{Int,N}; kwargs...) where {N} = Site(id; kwargs...)

Base.copy(x::Site) = x

"""
    id(site::Site)

Returns the coordinate location of the `site`.

See also: [`lanes`](@ref)
"""
function id end
id(site::Site{1}) = only(site.id)
id(site::Site) = site.id

Base.CartesianIndex(site::Site) = CartesianIndex(id(site))

"""
    isdual(site::Site)

Returns `true` if `site` is a dual site (i.e. is a "input"), `false` otherwise (i.e. is an "output").

See also: [`adjoint(::Site)`](@ref)
"""
isdual(site::Site) = site.dual
Base.show(io::IO, site::Site) = print(io, "$(id(site))$(site.dual ? "'" : "")")

"""
    adjoint(site::Site)

Returns the adjoint of `site`, i.e. a new `Site` object with the same coordinates as `site` but with the `dual` flag flipped (so an _input_ site becomes an _output_ site and vice versa).
"""
Base.adjoint(site::Site) = Site(id(site); dual=!site.dual)
Base.isless(a::Site, b::Site) = id(a) < id(b)

"""
    site"i,j,...[']"

Constructs a `Site` object with the given coordinates. The coordinates are given as a comma-separated list of integers. Optionally, a trailing `'` can be added to indicate that the site is a dual site (i.e. an "input").

See also: [`Site`](@ref)
"""
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

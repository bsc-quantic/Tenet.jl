abstract type AbstractLane end

"""
    Lane(id)
    Lane(i, j, ...)
    lane"i,j,..."

Represents the location of a physical index.

See also: [`Site`](@ref), [`lanes`](@ref)
"""
struct Lane{N} <: AbstractLane
    id::NTuple{N,Int}

    Lane(id::NTuple{N,Int}) where {N} = new{N}(id)
end

Lane(lane::Lane) = lane
Lane(id::Int) = Lane((id,))
Lane(id::Vararg{Int,N}) where {N} = Lane(id)
Lane(id::CartesianIndex) = Lane(Tuple(id))

Base.copy(x::Lane) = x

"""
    id(lane::AbstractLane)

Returns the coordinate location of the `lane`.

See also: [`lanes`](@ref)
"""
function id end
id(lane::Lane{1}) = only(lane.id)
id(lane::Lane) = lane.id
id(lane::AbstractLane) = id(Lane(lane))

Base.CartesianIndex(lane::AbstractLane) = CartesianIndex(id(lane))

Base.isless(a::AbstractLane, b::AbstractLane) = id(a) < id(b)

"""
    Site(id[; dual = false])
    Site(i, j, ...[; dual = false])
    site"i,j,...[']"

Represents a [`Lane`](@ref) with an annotation of input or output.
`Site` objects are used to label the indices of tensors in a [`Quantum`](@ref) Tensor Network.

See also: [`Lane`](@ref), [`sites`](@ref), [`isdual`](@ref)
"""
struct Site{N} <: AbstractLane
    lane::Lane{N}
    dual::Bool

    Site(lane::Lane{N}; dual=false) where {N} = new{N}(lane, dual)
end

Site(id::Int; kwargs...) = Site(Lane(id); kwargs...)
Site(@nospecialize(id::NTuple{N,Int}); kwargs...) where {N} = Site(Lane(id); kwargs...)
Site(@nospecialize(id::Vararg{Int,N}); kwargs...) where {N} = Site(Lane(id); kwargs...)
Site(@nospecialize(id::CartesianIndex); kwargs...) = Site(Lane(id); kwargs...)

Lane(site::Site) = site.lane

Base.copy(x::Site) = x

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
Base.adjoint(site::Site) = Site(id(site); dual=(!site.dual))

"""
    lane"i,j,..."

Constructs a `Lane` object with the given coordinates. The coordinates are given as a comma-separated list of integers.

See also: [`Lane`](@ref), [`@site_str`](@ref)
"""
macro lane_str(str)
    m = match(r"^(\d+,)*\d+$", str)
    isnothing(m) && error("Invalid site string: $str")

    id = tuple(map(eachmatch(r"(\d+)", str)) do match
        parse(Int, only(match.captures))
    end...)

    return :(Lane($id))
end

"""
    site"i,j,...[']"

Constructs a `Site` object with the given coordinates. The coordinates are given as a comma-separated list of integers. Optionally, a trailing `'` can be added to indicate that the site is a dual site (i.e. an "input").

See also: [`Site`](@ref), [`@lane_str`](@ref)
"""
macro site_str(str)
    m = match(r"^(\d+,)*\d+('?)$", str)
    isnothing(m) && error("Invalid site string: $str")

    id = tuple(map(eachmatch(r"(\d+)", str)) do match
        parse(Int, only(match.captures))
    end...)

    dual = endswith(str, "'")

    return :(Site($id; dual=($dual)))
end

Base.zero(x::Dict{Site,Symbol}) = x

struct Moment <: AbstractLane
    lane::Lane
    t::Int
end

Moment(lane::L, t) where {L<:AbstractLane} = Moment{L}(lane, t)

Lane(x::Moment) = Lane(x.lane)

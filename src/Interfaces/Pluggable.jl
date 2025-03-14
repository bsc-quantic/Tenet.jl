# This file defines the "Pluggable" interface; i.e. Tensor Networks that can be connected between each other.

struct PluggableInterface end

# this function should be used just for testing
function hasinterface(::PluggableInterface, T::Type)
    # required methods
    hasmethod(sites, Tuple{T}) || return false
    hasmethod(inds, Tuple{@NamedTuple{at::S} where {S<:Site},T}) || return false
    hasmethod(sites, Tuple{@NamedTuple{at::Symbol},T}) || return false
    return true
end

abstract type PluggableTrait end
struct IsPluggable end
struct WrapsPluggable end
struct NotPluggable end

function trait(::PluggableInterface, ::T) where {T}
    if hasmethod(unwrap, Tuple{PluggableInterface,T})
        return WrapsPluggable()
    elseif hasinterface(PluggableInterface(), T)
        return IsPluggable()
    else
        return NotPluggable()
    end
end

# required methods
"""
    sites(tn)

Returns the sites of the Tensor Network.
"""
sites(tn; kwargs...) = sites(sort_nt(values(kwargs)), tn)

sites(::@NamedTuple{}, tn) = sites((;), tn, trait(PluggableInterface(), tn))
sites(::@NamedTuple{}, tn, ::WrapsPluggable) = sites(unwrap(PluggableInterface(), tn))

"""
    inds(tn; at::Site)

Return the index linked to [`Site`](@ref) `at`.
"""
inds(kwargs::@NamedTuple{at::S}, tn) where {S<:Site} = inds(kwargs, tn, trait(PluggableInterface(), tn))
inds(kwargs::@NamedTuple{at::S}, tn, ::WrapsPluggable) where {S<:Site} = inds(kwargs, unwrap(PluggableInterface(), tn))

"""
    sites(tn; at::Symbol)

Return the site linked to index `at`.
"""
sites(kwargs::@NamedTuple{at::Symbol}, tn) = sites(kwargs, tn, trait(PluggableInterface(), tn))
sites(kwargs::@NamedTuple{at::Symbol}, tn, ::WrapsPluggable) = sites(kwargs, unwrap(PluggableInterface(), tn))

# optional methods
"""
    nsites(tn)

Return the number of sites of the Tensor Network.
"""
nsites(tn; kwargs...) = nsites(sort_nt(values(kwargs)), tn)
nsites(kwargs::NamedTuple, tn) = nsites(kwargs, tn, trait(PluggableInterface(), tn))

function nsites(kwargs::NamedTuple, tn, ::IsPluggable)
    @debug "Falling back to default implementation of `nsites(::$(typeof(kwargs)))`"
    return length(sites(kwargs, tn))
end

nsites(kwargs::NamedTuple, tn, ::WrapsPluggable) = nsites(kwargs, unwrap(PluggableInterface(), tn))

"""
    hassite(tn, s)

Return `true` if [`Site`](@ref) `s` is in the Tensor Network.
"""
hassite(tn, s::Site) = hassite(tn, s, trait(PluggableInterface(), tn))

function hassite(tn, s::Site, ::IsPluggable)
    @debug "Falling back to default implementation of `hassite`"
    s ∈ sites(tn)
end

hassite(tn, s::Site, ::WrapsPluggable) = hassite(unwrap(PluggableInterface(), tn), s)

# keyword methods
@valsplit sites(Val(kwargs::@NamedTuple{set::Symbol}), tn) = throw(ArgumentError("invalid `set` values: $(kwargs.set)"))
sites(::Val{(; set = :all)}, tn) = sites(tn)
sites(::Val{(; set = :inputs)}, tn) = sort!(filter(isdual, sites(tn)))
sites(::Val{(; set = :outputs)}, tn) = sort!(filter(!isdual, sites(tn)))

# mutating methods
"""
    addsite!(tn, site => ind)

Link `site` to `ind`.
"""
function addsite! end

addsite!(tn, p::Pair{<:Site,Symbol}) = addsite!(tn, p.first, p.second)
addsite!(tn, site::Site, ind::Symbol) = addsite!(tn, site, ind, trait(PluggableInterface(), tn))
addsite!(tn, site::Site, ind::Symbol, ::WrapsPluggable) = addsite!(unwrap(PluggableInterface(), tn), site, ind)

"""
    rmsite!(tn, site)

Unlink `site`.
"""
function rmsite! end

rmsite!(tn, site::Site) = rmsite!(tn, site, trait(PluggableInterface(), tn))
rmsite!(tn, site::Site, ::WrapsPluggable) = rmsite!(unwrap(PluggableInterface(), tn), site)

# derived methods
inds(::Val{(; set = :physical)}, tn) = [inds(tn; at=site) for site in sites(tn)]
inds(::Val{(; set = :virtual)}, tn) = setdiff(inds(tn), inds(tn; set=:physical))
inds(::Val{(; set = :inputs)}, tn) = [inds(tn; at=site) for site in sites(tn; set=:inputs)]
inds(::Val{(; set = :outputs)}, tn) = [inds(tn; at=site) for site in sites(tn; set=:outputs)]

# TODO commented out due to ambiguity error
# Base.in(s::Site, tn) = hassite(tn, s)

"""
    Socket

Abstract type representing the socket trait of a [`AbstractQuantum`](@ref) Tensor Network.
"""
abstract type Socket end

"""
    Scalar <: Socket

Socket representing a scalar; i.e. a Tensor Network with no open sites.
"""
struct Scalar <: Socket end

"""
    State <: Socket

Socket representing a state; i.e. a Tensor Network with only input sites (or only output sites if `dual = true`).
"""
@kwdef struct State <: Socket
    dual::Bool = false
end

"""
    Operator <: Socket

Socket representing an operator; i.e. a Tensor Network with both input and output sites.
"""
struct Operator <: Socket end

"""
    socket(q::AbstractQuantum)

Return the socket of a Tensor Network; i.e. whether it is a [`Scalar`](@ref), [`State`](@ref) or [`Operator`](@ref).
"""
function socket(tn)
    _sites = sites(tn)
    if isempty(_sites)
        Scalar()
    elseif all(!isdual, _sites)
        State()
    elseif all(isdual, _sites)
        State(; dual=true)
    else
        Operator()
    end
end

"""
    isconnectable(a, b)

Return `true` if two [Pluggable](@ref man-interface-pluggable) Tensor Networks can be connected. This means:

 1. The outputs of `a` are a superset of the inputs of `b`.
 2. The outputs of `a` and `b` are disjoint except for the sites that are connected.
"""
function isconnectable(a, b)
    Lane.(sites(a; set=:outputs)) ⊇ Lane.(sites(b; set=:inputs)) && isdisjoint(
        setdiff(Lane.(sites(a; set=:outputs)), Lane.(sites(b; set=:inputs))),
        setdiff(Lane.(sites(b; set=:inputs)), Lane.(sites(b; set=:outputs))),
    )
end

"""
    Base.adjoint(::AbstractTensorNetwork)

Return the adjoint of a Pluggable Tensor Network; i.e. the conjugate Tensor Network with the inputs and outputs swapped.
"""
Base.adjoint(tn::AbstractTensorNetwork) = adjoint_sites!(conj(tn))

"""
    LinearAlgebra.adjoint!(::AbstractTensorNetwork)

Like [`adjoint`](@ref), but in-place.
"""
LinearAlgebra.adjoint!(tn::AbstractTensorNetwork) = adjoint_sites!(conj!(tn))

# update site information and rename inner indices
function adjoint_sites!(tn)
    # generate mapping
    mapping = Dict(site => inds(tn; at=site) for site in sites(tn))

    # remove sites preemptively to avoid issues on renaming
    for site in sites(tn)
        rmsite!(tn, site)
    end

    # set new site mapping
    for (site, index) in mapping
        addsite!(tn, site' => index)
    end

    # rename inner indices
    # replace!(tn, map(i -> i => Symbol(i, "'"), inds(tn; set=:virtual)))

    return tn
end

"""
    align!(a, ioa, b, iob)

Align the physical indices of `b` to match the physical indices of `a`. `ioa` and `iob` are either `:inputs` or `:outputs`.
"""
function align!(a, ioa, b, iob)
    @assert ioa === :inputs || ioa === :outputs
    @assert iob === :inputs || iob === :outputs

    # If `reset=true`, then all indices are renamed. If `reset=false`, then only the indices of the input/output sites are renamed.

    # if !isdisjoint(inds(a), inds(b))
    #     @warn "Overlapping indices"
    # end

    # if reset
    #     @debug "[align!] Renaming indices of b"
    #     resetinds!(b, :gensymclean)
    # end

    targets = Lane.(sites(a; set=ioa)) ∩ Lane.(sites(b; set=iob))

    target_sites_a = Site.(targets; dual=ioa === :inputs)
    target_sites_b = Site.(targets; dual=iob === :inputs)

    replacements = map(zip(target_sites_a, target_sites_b)) do (sitea, siteb)
        inds(b; at=siteb) => inds(a; at=sitea)
    end

    if issetequal(first.(replacements), last.(replacements))
        return b
    end

    replace!(b, replacements)

    return a, b
end

align!((a, b)::P) where {P<:Pair} = align!(a, :outputs, b, :inputs)

"""
    @align! a => b reset=true

Rename in-place the indices of the input/output sites of two Pluggable Tensor Networks to be able to connect between them.
"""
macro align!(expr)
    @assert Meta.isexpr(expr, :call) && expr.args[1] == :(=>)
    Base.remove_linenums!(expr)
    a, b = expr.args[2:end]

    # @assert Meta.isexpr(reset, :(=)) && reset.args[1] == :reset

    @assert Meta.isexpr(a, :call)
    @assert Meta.isexpr(b, :call)
    ioa, ida = a.args
    iob, idb = b.args
    return quote
        align!($(esc(ida)), $(Meta.quot(ioa)), $(esc(idb)), $(Meta.quot(iob)))
        $(esc(idb))
    end
end

@deprecate inputs(tn) sites(tn; set=:inputs)
@deprecate outputs(tn) sites(tn; set=:outputs)
@deprecate ninputs(tn) nsites(tn; set=:inputs)
@deprecate noutputs(tn) nsites(tn; set=:outputs)

@deprecate reindex!(args...; kwargs...) align!(args...; kwargs...)

macro reindex!(args...)
    Base.depwarn("Macro @reindex! is deprecated, use @align! instead", :@align!)
    :(@reindex!($(args...)))
end

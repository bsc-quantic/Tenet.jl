# This file defines the "Pluggable" interface; i.e. Tensor Networks that can be connected between each other.

"""
    sites(tn)

Return the sites of the Tensor Network.
"""
sites(tn::AbstractTensorNetwork; kwargs...) = sites(sort_nt(values(kwargs)), tn)

"""
    sites(tn;)

Returns the sites of the Tensor Network.

!!! note

    This is the method called by `sites(tn)` when no kwarg are passed.
"""
sites(::@NamedTuple{}, tn::AbstractTensorNetwork)

"""
    inds(tn; at::Site)

Return the index linked to [`Site`](@ref) `at`.

!!! note

    This is the method called by `inds(tn; at=site)`.
"""
inds(::@NamedTuple{at::S}, ::AbstractTensorNetwork) where {S<:Site}

"""
    sites(tn; at::Symbol)

Return the site linked to index `at`.

!!! note

    This is the method called by `inds(tn; at=site)`.
"""
sites(::@NamedTuple{at::Symbol}, ::AbstractTensorNetwork)

# optional methods
"""
    nsites(tn)

Return the number of sites of the Tensor Network.
"""
nsites(tn; kwargs...) = length(sites(tn; kwargs...))

hassite(tn::AbstractTensorNetwork, s::Site) = s ∈ sites(tn)
Base.in(s::Site, tn::AbstractTensorNetwork) = s ∈ sites(tn)

# keyword methods
function sites(kwargs::@NamedTuple{plugset::Symbol}, tn::AbstractTensorNetwork)
    if kwargs.plugset === :inputs
        sort!(filter(isdual, sites(tn)))
    elseif kwargs.plugset === :outputs
        sort!(filter(!isdual, sites(tn)))
    else
        throw(ArgumentError("invalid `plugset` values: $(kwargs.plugset)"))
    end
end

# mutating methods
"""
    addsite!(tn, site => ind)

Register mapping `site` to `ind`.
"""
function addsite! end

"""
    rmsite!(tn, site)

Unregister `site`.
"""
function rmsite! end

# derived methods
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
function socket(tn::AbstractTensorNetwork)
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
function adjoint_sites!(tn::AbstractTensorNetwork)
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
    resetinds!(tn::AbstractTensorNetwork, method=:gensymnew; kwargs...)

Rename indices in the `TensorNetwork` to a new set of indices. It is mainly used to avoid index name conflicts when connecting Tensor Networks.
"""
function resetinds!(tn::AbstractTensorNetwork, method=:gensym; kwargs...)
    new_name_f = if method === :suffix
        (ind) -> Symbol(ind, get(kwargs, :suffix, '\''))
    elseif method === :gensymwrap
        (ind) -> gensym(ind)
    elseif methods === :gensymnew
        (_) -> gensym(get(kwargs, :base, :i))
    elseif method === :characters
        gen = IndexCounter(get(kwargs, :init, 1))
        (_) -> nextindex!(gen)
    else
        error("Invalid method: $(Meta.quot(method))")
    end

    _inds = if haskey(kwargs, :plugset)
        inds(tn; plugset=kwargs.plugset)
    else
        inds(tn)
    end

    for ind in _inds
        replace!(tn, ind => new_name_f(ind))
    end
end

"""
    align!(a, ioa, b, iob)

Align the physical indices of `b` to match the physical indices of `a`. `ioa` and `iob` are either `:inputs` or `:outputs`.
"""
function align!(a, ioa, b, iob)
    targets = Lane.(sites(a; set=ioa)) ∩ Lane.(sites(b; set=iob))

    target_sites_a = Site.(targets; dual=ioa === :inputs)
    target_sites_b = Site.(targets; dual=iob === :inputs)

    replacements = map(zip(target_sites_a, target_sites_b)) do sitea, siteb
        inds(b; at=siteb) => inds(a; at=sitea)
    end

    if issetequal(first.(replacements), last.(replacements))
        return b
    end

    replace!(b, replacements)

    return a, b
end

align!((a, b)::Pair) = align!(a, :outputs, b, :inputs)

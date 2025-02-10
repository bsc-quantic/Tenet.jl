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

# optional methods
"""
    nsites(tn)

Return the number of sites of the Tensor Network.
"""
nsites(tn; kwargs...) = length(sites(tn; kwargs...))

hassite(tn::AbstractTensorNetwork, s::Site) = s ∈ sites(tn)
Base.in(s::Site, tn::AbstractTensorNetwork) = s ∈ sites(tn)

# derived methods
function sites(kwargs::@NamedTuple{plugs::Symbol}, tn::AbstractTensorNetwork)
    if kwargs.plugs === :inputs
        sort!(filter(isdual, sites(tn)))
    elseif kwargs.plugs === :outputs
        sort!(filter(!isdual, sites(tn)))
    else
        throw(ArgumentError("invalid `plugs` values: $(kwargs.plugs)"))
    end
end

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

# TODO
"""
    @reindex! a => b reset=true

Rename in-place the indices of the input/output sites of two [`Quantum`](@ref) Tensor Networks to be able to connect between them.
If `reset=true`, then all indices are renamed. If `reset=false`, then only the indices of the input/output sites are renamed.
"""
macro reindex!(expr, reset=:(reset = true))
    # @assert Meta.isexpr(expr, :call) && expr.args[1] == :(=>)
    # Base.remove_linenums!(expr)
    # a, b = expr.args[2:end]

    # @assert Meta.isexpr(reset, :(=)) && reset.args[1] == :reset

    # @assert Meta.isexpr(a, :call)
    # @assert Meta.isexpr(b, :call)
    # ioa, ida = a.args
    # iob, idb = b.args
    # return quote
    #     reindex!(Quantum($(esc(ida))), $(Meta.quot(ioa)), Quantum($(esc(idb))), $(Meta.quot(iob)); $(esc(reset)))
    #     $(esc(idb))
    # end
end

function reindex!(a, ioa, b, iob; reset=true)
    if reset
        resetinds!(a)
        resetinds!(b; init=ninds(a) + 1)
    end

    sitesa = if ioa === :inputs
        collect(sites(a; plugs=:inputs))
    elseif ioa === :outputs
        collect(sites(a; plugs=:outputs))
    else
        error("Invalid argument: $(Meta.quot(ioa))")
    end

    sitesb = if iob === :inputs
        collect(sites(b; plugs=:inputs))
    elseif iob === :outputs
        collect(sites(b; plugs=:outputs))
    else
        error("Invalid argument: :$(Meta.quot(iob))")
    end

    # TODO select sites to reindex
    targetsites = (ioa === :inputs ? adjoint.(sitesa) : sitesa) ∩ (iob === :inputs ? adjoint.(sitesb) : sitesb)

    replacements = map(targetsites) do site
        siteb = iob === :inputs ? site' : site
        sitea = ioa === :inputs ? site' : site
        inds(b; at=siteb) => inds(a; at=sitea)
    end

    if issetequal(first.(replacements), last.(replacements))
        return b
    end

    replace!(b, replacements)

    return b
end

# TODO
# function resetinds!(tn; init=1)
#     qtn = Quantum(tn)

#     mapping = resetinds!(Val(:return_mapping), tn; init)
#     replace!(TensorNetwork(qtn), mapping)

#     for (site, index) in qtn.sites
#         qtn.sites[site] = mapping[index]
#     end

#     return tn
# end
# resetinds(tn; init=1) = resetinds!(copy(tn); init)

"""
    Base.adjoint(::AbstractQuantum)

Return the adjoint of a [`Quantum`](@ref) Tensor Network; i.e. the conjugate Tensor Network with the inputs and outputs swapped.
"""
# Base.adjoint(tn::AbstractQuantum) = adjoint_sites!(conj(tn))

"""
    LinearAlgebra.adjoint!(::AbstractQuantum)

Like [`adjoint`](@ref), but in-place.
"""
# LinearAlgebra.adjoint!(tn::AbstractQuantum) = adjoint_sites!(conj!(tn))

# update site information and rename inner indices
# function adjoint_sites!(tn::AbstractQuantum)
#     oldsites = copy(Quantum(tn).sites)
#     empty!(Quantum(tn).sites)
#     for (site, index) in oldsites
#         addsite!(tn, site', index)
#     end

#     # rename inner indices
#     replace!(tn, map(i -> i => Symbol(i, "'"), inds(tn; set=:virtual)))

#     return tn
# end

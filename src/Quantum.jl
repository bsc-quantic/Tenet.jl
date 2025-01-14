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
    AbstractQuantum

Abstract type for `Quantum`-derived types.
Its subtypes must implement conversion or extraction of the underlying `Quantum` by overloading the `Quantum` constructor.
"""
abstract type AbstractQuantum <: AbstractTensorNetwork end

"""
    Quantum

Tensor Network with a notion of "causality". This leads to the concept of sites and directionality (input/output).

# Notes

  - Indices are referenced by `Site`s.
"""
struct Quantum <: AbstractQuantum
    tn::TensorNetwork

    # WARN keep them synchronized
    sites::Dict{Site,Symbol}
    # sitetensors::Dict{Site,Tensor}

    function Quantum(tn::TensorNetwork, sites)
        for (_, index) in sites
            if !haskey(tn.indexmap, index)
                error("Index $index not found in TensorNetwork")
            elseif index ∉ inds(tn; set=:open)
                error("Index $index must be open")
            end
        end

        # sitetensors = map(sites) do (site, index)
        #     site => tn[index]
        # end |> Dict{Site,Tensor}

        return new(tn, sites)
    end
end

Quantum(tn::Quantum) = tn

"""
    Quantum(array, sites)

Construct a [`Quantum`](@ref) Tensor Network from an array and a list of sites. Useful for simple operators like gates.
"""
function Quantum(array, sites)
    if ndims(array) != length(sites)
        throw(ArgumentError("Number of sites must match number of dimensions of array"))
    end

    gen = IndexCounter()
    symbols = map(_ -> nextindex!(gen), sites)
    sitemap = Dict{Site,Symbol}(
        map(sites, 1:ndims(array)) do site, i
            site => symbols[i]
        end,
    )
    tensor = Tensor(array, symbols)
    tn = TensorNetwork([tensor])
    qtn = Quantum(tn, sitemap)
    return qtn
end

"""
    TensorNetwork(q::AbstractQuantum)

Return the underlying `TensorNetwork` of an [`AbstractQuantum`](@ref).
"""
TensorNetwork(tn::AbstractQuantum) = Quantum(tn).tn

Base.copy(tn::Quantum) = Quantum(copy(TensorNetwork(tn)), copy(tn.sites))

Base.similar(tn::Quantum) = Quantum(similar(TensorNetwork(tn)), copy(tn.sites))
Base.zero(tn::Quantum) = Quantum(zero(TensorNetwork(tn)), copy(tn.sites))

function Base.:(==)(a::AbstractQuantum, b::AbstractQuantum)
    return Quantum(a).sites == Quantum(b).sites && @invoke ==(a::AbstractTensorNetwork, b::AbstractTensorNetwork)
end
function Base.isapprox(a::AbstractQuantum, b::AbstractQuantum; kwargs...)
    return Quantum(a).sites == Quantum(b).sites &&
           @invoke isapprox(a::AbstractTensorNetwork, b::AbstractTensorNetwork; kwargs...)
end

Base.summary(io::IO, tn::AbstractQuantum) = print(io, "$(ntensors(tn))-tensors Quantum")
function Base.show(io::IO, tn::T) where {T<:AbstractQuantum}
    return print(io, "$T (inputs=$(nsites(tn; set=:inputs)), outputs=$(nsites(tn; set=:outputs)))")
end

tensors(kwargs::NamedTuple{(:at,)}, tn::AbstractQuantum) = only(tensors(tn; intersects=inds(tn; at=kwargs.at)))
inds(kwargs::NamedTuple{(:at,)}, tn::AbstractQuantum) = Quantum(tn).sites[kwargs.at]

function inds(kwargs::NamedTuple{(:set,)}, tn::AbstractQuantum)
    if kwargs.set === :physical
        return map(sites(tn)) do site
            inds(tn; at=site)::Symbol
        end
    elseif kwargs.set === :virtual
        return setdiff(inds(tn), inds(tn; set=:physical))
    elseif kwargs.set ∈ (:inputs, :outputs)
        return map(sites(tn; kwargs.set)) do site
            inds(tn; at=site)::Symbol
        end
    else
        return inds(TensorNetwork(tn); set=kwargs.set)
    end
end

@deprecate Base.getindex(q::Quantum, site::Site) inds(q; at=site) false

# `pop!` / `delete!` methods call this method
function Base.pop!(tn::AbstractQuantum, tensor::Tensor)
    @invoke pop!(tn::AbstractTensorNetwork, tensor)

    # TODO replace with `inds(tn; set=:physical)` when implemented
    targets = values(Quantum(tn).sites) ∩ inds(tensor)
    for target in targets
        rmsite!(tn, findfirst(==(target), Quantum(tn).sites))
    end

    return tensor
end

function Base.replace!(tn::AbstractQuantum, old_new::Pair{Symbol,Symbol})
    tn = Quantum(tn)

    # replace indices in underlying Tensor Network
    @invoke replace!(tn::AbstractTensorNetwork, old_new)

    # replace indices in site information
    site = sites(tn; at=first(old_new))
    if !isnothing(site)
        rmsite!(tn, site)
        addsite!(tn, site, last(old_new))
    end

    return tn
end

# FIXME return type should be the original type, not `Quantum`
function Base.replace!(tn::AbstractQuantum, old_new::Base.AbstractVecOrTuple{Pair{Symbol,Symbol}})
    tn = Quantum(tn)

    # replace indices in underlying Tensor Network
    @invoke replace!(tn::AbstractTensorNetwork, old_new)

    # replace indices in site information
    from, to = first.(old_new), last.(old_new)
    for (site, index) in tn.sites
        i = findfirst(==(index), from)
        if !isnothing(i)
            tn.sites[site] = to[i]
        end
    end

    return tn
end

function reindex!(a::Quantum, ioa, b::Quantum, iob; reset=true)
    if reset
        resetinds!(a)
        resetinds!(b; init=ninds(TensorNetwork(a)) + 1)
    end

    sitesa = if ioa === :inputs
        collect(sites(a; set=:inputs))
    elseif ioa === :outputs
        collect(sites(a; set=:outputs))
    else
        error("Invalid argument: $(Meta.quot(ioa))")
    end

    sitesb = if iob === :inputs
        collect(sites(b; set=:inputs))
    elseif iob === :outputs
        collect(sites(b; set=:outputs))
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

function resetinds!(tn::AbstractQuantum; init=1)
    qtn = Quantum(tn)

    mapping = resetinds!(Val(:return_mapping), tn; init)
    replace!(TensorNetwork(qtn), mapping)

    for (site, index) in qtn.sites
        qtn.sites[site] = mapping[index]
    end

    return tn
end
resetinds(tn::AbstractQuantum; init=1) = resetinds!(copy(tn); init)

"""
    @reindex! a => b reset=true

Rename in-place the indices of the input/output sites of two [`Quantum`](@ref) Tensor Networks to be able to connect between them.
If `reset=true`, then all indices are renamed. If `reset=false`, then only the indices of the input/output sites are renamed.
"""
macro reindex!(expr, reset=:(reset = true))
    @assert Meta.isexpr(expr, :call) && expr.args[1] == :(=>)
    Base.remove_linenums!(expr)
    a, b = expr.args[2:end]

    @assert Meta.isexpr(reset, :(=)) && reset.args[1] == :reset

    @assert Meta.isexpr(a, :call)
    @assert Meta.isexpr(b, :call)
    ioa, ida = a.args
    iob, idb = b.args
    return quote
        reindex!(Quantum($(esc(ida))), $(Meta.quot(ioa)), Quantum($(esc(idb))), $(Meta.quot(iob)); $(esc(reset)))
        $(esc(idb))
    end
end

"""
    sites(q::AbstractQuantum)

Return the sites of a [`AbstractQuantum`](@ref) Tensor Network.
"""
function sites end

sites(tn::AbstractQuantum; kwargs...) = sites(sort_nt(values(kwargs)), tn)
sites(::@NamedTuple{}, tn::AbstractQuantum) = sites((; set=:all), tn)

"""
    nsites(q::AbstractQuantum)

Return the number of sites of a [`AbstractQuantum`](@ref) Tensor Network.
"""
nsites(tn::AbstractQuantum; kwargs...) = length(sites(tn; kwargs...))

@deprecate inputs(tn::AbstractQuantum) sites(tn; set=:inputs)
@deprecate outputs(tn::AbstractQuantum) sites(tn; set=:outputs)
@deprecate ninputs(tn::AbstractQuantum) nsites(tn; set=:inputs)
@deprecate noutputs(tn::AbstractQuantum) nsites(tn; set=:outputs)

"""
    lanes(q::AbstractQuantum)

Return the lanes of a [`AbstractQuantum`](@ref) Tensor Network.
"""
lanes(tn::AbstractQuantum) = unique!(Lane[Lane.(sites(tn; set=:inputs))..., Lane.(sites(tn; set=:outputs))...])

"""
    nlanes(q::AbstractQuantum)

Return the number of lanes of a [`AbstractQuantum`](@ref) Tensor Network.
"""
nlanes(tn::AbstractQuantum) = length(lanes(tn))

function addsite!(tn::AbstractQuantum, site, index)
    tn = Quantum(tn)
    if haskey(tn.sites, site)
        error("Site $site already exists")
    end

    if index ∉ inds(tn; set=:open)
        error("Index $index must be open")
    end

    return tn.sites[site] = index
end

function rmsite!(tn::AbstractQuantum, site)
    tn = Quantum(tn)
    if !haskey(tn.sites, site)
        error("Site $site does not exist")
    end

    return delete!(tn.sites, site)
end

hassite(tn::AbstractQuantum, site) = haskey(Quantum(tn).sites, site)
Base.in(site::Site, tn::AbstractQuantum) = hassite(tn, site)

function sites(kwargs::NamedTuple{(:set,)}, tn::AbstractQuantum)
    tn = Quantum(tn)
    if kwargs.set === :all
        sort!(collect(keys(tn.sites)))
    elseif kwargs.set === :inputs
        sort!(collect(Iterators.filter(isdual, keys(tn.sites))))
    elseif kwargs.set === :outputs
        sort!(collect(Iterators.filter(!isdual, keys(tn.sites))))
    else
        throw(ArgumentError("invalid set: $(kwargs.set)"))
    end
end

function sites(kwargs::@NamedTuple{at::Symbol}, tn::AbstractQuantum)
    tn = Quantum(tn)
    return findfirst(==(kwargs.at), tn.sites)
end

"""
    isconnectable(a::AbstractQuantum, b::AbstractQuantum)

Return `true` if two [`AbstractQuantum`](@ref) Tensor Networks can be connected. This means:

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
    socket(q::AbstractQuantum)

Return the socket of a [`Quantum`](@ref) Tensor Network; i.e. whether it is a [`Scalar`](@ref), [`State`](@ref) or [`Operator`](@ref).
"""
function socket(q::AbstractQuantum)
    _sites = sites(q)
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
    Base.adjoint(::AbstractQuantum)

Return the adjoint of a [`Quantum`](@ref) Tensor Network; i.e. the conjugate Tensor Network with the inputs and outputs swapped.
"""
Base.adjoint(tn::AbstractQuantum) = adjoint_sites!(conj(tn))

"""
    LinearAlgebra.adjoint!(::AbstractQuantum)

Like [`adjoint`](@ref), but in-place.
"""
LinearAlgebra.adjoint!(tn::AbstractQuantum) = adjoint_sites!(conj!(tn))

# update site information and rename inner indices
function adjoint_sites!(tn::AbstractQuantum)
    oldsites = copy(Quantum(tn).sites)
    empty!(Quantum(tn).sites)
    for (site, index) in oldsites
        addsite!(tn, site', index)
    end

    # rename inner indices
    replace!(tn, map(i -> i => Symbol(i, "'"), inds(tn; set=:virtual)))

    return tn
end

"""
    Base.merge(a::AbstractQuantum, b::AbstractQuantum; reset=true)

Merge multiple [`AbstractQuantum`](@ref) Tensor Networks. If `reset=true`, then all indices are renamed. If `reset=false`, then only the indices of the input/output sites are renamed.

See also: [`merge!`](@ref), [`@reindex!`](@ref).
"""
Base.merge(tns::AbstractQuantum...; kwargs...) = foldl((a, b) -> merge!(a, b; kwargs...), copy.(tns))
Base.merge!(tns::AbstractQuantum...; kwargs...) = foldl((a, b) -> merge!(a, b; kwargs...), tns)

"""
    Base.merge!(::AbstractQuantum...; reset=true)

Merge in-place multiple [`AbstractQuantum`](@ref) Tensor Networks. If `reset=true`, then all indices are renamed. If `reset=false`, then only the indices of the input/output sites are renamed.

See also: [`merge`](@ref), [`@reindex!`](@ref).
"""
function Base.merge!(a::AbstractQuantum, b::AbstractQuantum; reset=true)
    @assert adjoint.(sites(b; set=:inputs)) ⊆ sites(a; set=:outputs) "Inputs of b must match outputs of a"
    @assert isdisjoint(setdiff(sites(b; set=:outputs), adjoint.(sites(b; set=:inputs))), sites(a; set=:outputs)) "b cannot create new sites where is not connected"

    @reindex! outputs(a) => inputs(b) reset = reset
    merge!(TensorNetwork(a), TensorNetwork(b))

    for site in sites(b; set=:inputs)
        rmsite!(a, site')
    end

    for site in sites(b; set=:outputs)
        addsite!(a, site, inds(b; at=site))
    end

    return a
end

# NOTE do not document because we might move it down to `Ansatz`
LinearAlgebra.normalize(ψ::AbstractQuantum; kwargs...) = normalize!(copy(ψ); kwargs...)

"""
    LinearAlgebra.norm(::AbstractQuantum, p=2; kwargs...)

Return the Lp-norm of a [`AbstractQuantum`](@ref) Tensor Network.

!!! warning

    Only L2-norm is implemented yet.
"""
function LinearAlgebra.norm(ψ::AbstractQuantum, p::Real=2; kwargs...)
    p == 2 || throw(ArgumentError("only L2-norm is implemented yet"))
    return LinearAlgebra.norm2(ψ; kwargs...)
end

LinearAlgebra.norm2(ψ::AbstractQuantum; kwargs...) = LinearAlgebra.norm2(socket(ψ), ψ; kwargs...)

function LinearAlgebra.norm2(::State, ψ::AbstractQuantum; kwargs...)
    return abs(sqrt(only(contract(merge(ψ, ψ'); kwargs...))))
end

function LinearAlgebra.norm2(::Operator, ψ::AbstractQuantum; kwargs...)
    ψ, ϕ = Quantum(ψ), Quantum(ψ')

    @reindex! outputs(ψ) => inputs(ϕ) reset = false
    @reindex! inputs(ψ) => outputs(ϕ) reset = false
    return abs(sqrt(only(contract(merge(TensorNetwork(ψ), TensorNetwork(ϕ)); kwargs...))))
end

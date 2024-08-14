"""
    AbstractQuantum

Abstract type for `Quantum`-derived types.
Its subtypes must implement conversion or extraction of the underlying `Quantum` by overloading the `Quantum` constructor.
"""
abstract type AbstractQuantum <: AbstractTensorNetwork end

# `AbstractTensorNetwork` interface
TensorNetwork(tn::AbstractQuantum) = TensorNetwork(Quantum(tn))

@kwmethod tensors(tn::AbstractQuantum; at) = only(tensors(tn; intersects=inds(tn; at)))

# `pop!` / `delete!` methods call this method
function Base.pop!(tn::AbstractQuantum, tensor)
    @invoke pop!(tn::AbstractTensorNetwork, tensor)

    # TODO replace with `inds(tn; set=:physical)` when implemented
    targets = values(Quantum(tn).sites) ∩ inds(tensor)
    for target in targets
        rmsite!(tn, findfirst(==(target), Quantum(tn).sites))
    end

    return tensor
end

# `AbstractQuantum` interface
addsite!(tn::AbstractQuantum, site, index) = addsite!(Quantum(tn), site, index)
rmsite!(tn::AbstractQuantum, site) = rmsite!(Quantum(tn), site)

"""
    sites(q::AbstractQuantum)

Returns the sites of a [`AbstractQuantum`](@ref) Tensor Network.
"""
function sites end

@kwdispatch sites(tn::AbstractQuantum)
@kwmethod sites(tn::AbstractQuantum;) = sites(tn; set=:all)

"""
    nsites(q::AbstractQuantum)

Returns the number of sites of a [`AbstractQuantum`](@ref) Tensor Network.
"""
nsites(tn::AbstractQuantum; kwargs...) = length(sites(tn; kwargs...))

@deprecate inputs(tn::AbstractQuantum) sites(tn; set=:inputs)
@deprecate outputs(tn::AbstractQuantum) sites(tn; set=:outputs)
@deprecate ninputs(tn::AbstractQuantum) nsites(tn; set=:inputs)
@deprecate noutputs(tn::AbstractQuantum) nsites(tn; set=:outputs)

"""
    lanes(q::AbstractQuantum)

Returns the lanes of a [`AbstractQuantum`](@ref) Tensor Network.
"""
function lanes(tn::AbstractQuantum)
    return unique(
        Iterators.map(Iterators.flatten([sites(tn; set=:inputs), sites(tn; set=:outputs)])) do site
            isdual(site) ? site' : site
        end,
    )
end

"""
    nlanes(q::AbstractQuantum)

Returns the number of lanes of a [`AbstractQuantum`](@ref) Tensor Network.
"""
nlanes(tn::AbstractQuantum) = length(lanes(tn))

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
    socket(q::Quantum)

Returns the socket of a [`Quantum`](@ref) Tensor Network; i.e. whether it is a [`Scalar`](@ref), [`State`](@ref) or [`Operator`](@ref).
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
    Quantum

Tensor Network with a notion of "causality". This leads to the notion of sites and directionality (input/output).

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

Quantum(qtn::Quantum) = qtn

"""
    TensorNetwork(q::Quantum)

Returns the underlying `TensorNetwork` of a [`Quantum`](@ref) Tensor Network.
"""
TensorNetwork(q::Quantum) = q.tn

Base.copy(q::Quantum) = Quantum(copy(TensorNetwork(q)), copy(q.sites))

Base.similar(q::Quantum) = Quantum(similar(TensorNetwork(q)), copy(q.sites))
Base.zero(q::Quantum) = Quantum(zero(TensorNetwork(q)), copy(q.sites))

Base.:(==)(a::Quantum, b::Quantum) = a.tn == b.tn && a.sites == b.sites
Base.isapprox(a::Quantum, b::Quantum; kwargs...) = isapprox(a.tn, b.tn; kwargs...) && a.sites == b.sites

Base.summary(io::IO, q::Quantum) = print(io, "$(length(q.tn.tensormap))-tensors Quantum")
function Base.show(io::IO, q::Quantum)
    return print(io, "Quantum (inputs=$(nsites(q; set=:inputs)), outputs=$(nsites(q; set=:outputs)))")
end

@kwmethod inds(tn::AbstractQuantum; at) = Quantum(tn).sites[at]

"""
    adjoint(q::Quantum)

Returns the adjoint of a [`Quantum`](@ref) Tensor Network; i.e. the conjugate Tensor Network with the inputs and outputs swapped.
"""
function Base.adjoint(qtn::Quantum)
    sites = Dict{Site,Symbol}(
        Iterators.map(qtn.sites) do (site, index)
            site' => index
        end,
    )

    tn = conj(qtn)

    # rename inner indices
    physical_inds = values(sites)
    virtual_inds = setdiff(inds(tn), physical_inds)
    replace!(tn, map(virtual_inds) do i
        i => Symbol(i, "'")
    end...)

    return Quantum(TensorNetwork(tn), sites)
end

function addsite!(tn::Quantum, site, index)
    if haskey(tn.sites, site)
        error("Site $site already exists")
    end

    if index ∉ inds(tn; set=:open)
        error("Index $index must be open")
    end

    return tn.sites[site] = index
end

function rmsite!(tn::Quantum, site)
    if !haskey(tn.sites, site)
        error("Site $site does not exist")
    end

    return delete!(tn.sites, site)
end

@kwmethod function sites(tn::AbstractQuantum; set)
    tn = Quantum(tn)
    if set === :all
        sort!(collect(keys(tn.sites)))
    elseif set === :inputs
        sort!(filter(isdual, keys(tn.sites)))
    elseif set === :outputs
        sort!(filter(!isdual, keys(tn.sites)))
    else
        throw(ArgumentError("invalid set: $set"))
    end
end

@deprecate Base.getindex(q::Quantum, site::Site) inds(q; at=site) false

function Base.replace!(tn::Quantum, old_new::Base.AbstractVecOrTuple{Pair{Symbol,Symbol}})
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

function reindex!(a::Quantum, ioa, b::Quantum, iob)
    ioa ∈ [:inputs, :outputs] || error("Invalid argument: :$ioa")

    resetindex!(a)
    resetindex!(b; init=ninds(TensorNetwork(a)) + 1)

    sitesb = if iob === :inputs
        collect(sites(b; set=:inputs))
    elseif iob === :outputs
        collect(sites(b; set=:outputs))
    else
        error("Invalid argument: :$iob")
    end

    replacements = map(sitesb) do site
        inds(b; at=site) => inds(a; at=ioa != iob ? site' : site)
    end

    if issetequal(first.(replacements), last.(replacements))
        return b
    end

    replace!(b, replacements)

    return b
end

function resetindex!(tn::AbstractQuantum; init=1)
    tn = Quantum(tn)

    mapping = resetindex!(Val(:return_mapping), tn; init)
    replace!(TensorNetwork(tn), mapping)

    for (site, index) in tn.sites
        tn.sites[site] = mapping[index]
    end
end

"""
    @reindex! a => b

Reindexes the input/output sites of two [`Quantum`](@ref) Tensor Networks to be able to connect between them.
"""
macro reindex!(expr)
    @assert Meta.isexpr(expr, :call) && expr.args[1] == :(=>)
    Base.remove_linenums!(expr)
    a, b = expr.args[2:end]

    @assert Meta.isexpr(a, :call)
    @assert Meta.isexpr(b, :call)
    ioa, ida = a.args
    iob, idb = b.args
    return :((reindex!(Quantum($(esc(ida))), $(Meta.quot(ioa)), Quantum($(esc(idb))), $(Meta.quot(iob)))); $(esc(idb)))
end

"""
    merge(a::Quantum, b::Quantum...)

Merges multiple [`Quantum`](@ref) Tensor Networks into a single one by connecting input/output sites.
"""
Base.merge(a::Quantum, others::Quantum...) = foldl(merge, others; init=a)
function Base.merge(a::Quantum, b::Quantum)
    @assert issetequal(sites(a; set=:outputs), map(adjoint, sites(b; set=:inputs))) "Outputs of $a must match inputs of $b"

    @reindex! outputs(a) => inputs(b)
    tn = merge(TensorNetwork(a), TensorNetwork(b))

    sites = Dict{Site,Symbol}()

    for site in sites(a; set=:inputs)
        sites[site] = inds(a; at=site)
    end

    for site in sites(b; set=:outputs)
        sites[site] = inds(b; at=site)
    end

    return Quantum(tn, sites)
end

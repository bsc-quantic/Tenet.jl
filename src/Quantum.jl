"""
    AbstractQuantum

Abstract type for `Quantum`-derived types.
Its subtypes must implement conversion or extraction of the underlying `Quantum` by overloading the `Quantum` constructor.
"""
abstract type AbstractQuantum <: AbstractTensorNetwork end

# `AbstractTensorNetwork` interface
TensorNetwork(tn::AbstractQuantum) = TensorNetwork(Quantum(tn))

inds(tn::AbstractQuantum, ::Val{:at}, site::Site) = inds(Quantum(tn), Val(:at), site)
tensors(tn::AbstractQuantum, ::Val{:at}, site::Site) = only(tensors(tn; intersects=inds(tn; at=site)))

# `AbstractQuantum` interface
# TODO would be simpler and easier by overloading `Core.kwcall`? ⚠️ it's an internal implementation detail
"""
    sites(q::AbstractQuantum)

Returns the sites of a [`AbstractQuantum`](@ref) Tensor Network.
"""
function sites(tn::AbstractQuantum; kwargs...)
    isempty(kwargs) && return sites(tn, Val(:set), :all)
    key = only(keys(kwargs))
    value = values(kwargs)[key]
    return sites(tn, Val(key), value)
end

nsites(tn::AbstractQuantum; kwargs...) = nsites(Quantum(tn); kwargs...)

"""
    inputs(q::Quantum)

Returns the input sites of a [`Quantum`](@ref) Tensor Network.
"""
# inputs(q::Quantum) = sort!(collect(filter(isdual, keys(q.sites))))
@deprecate inputs(tn::AbstractQuantum) sites(tn; set=:inputs)

"""
    outputs(q::Quantum)

Returns the output sites of a [`Quantum`](@ref) Tensor Network.
"""
# outputs(q::Quantum) = sort!(collect(filter(!isdual, keys(q.sites))))
@deprecate outputs(tn::AbstractQuantum) sites(tn; set=:outputs)

"""
    ninputs(q::Quantum)

Returns the number of input sites of a [`Quantum`](@ref) Tensor Network.
"""
# ninputs(q::Quantum) = count(isdual, keys(q.sites))
@deprecate ninputs(tn::AbstractQuantum) nsites(tn; set=:inputs)

"""
    noutputs(q::Quantum)

Returns the number of output sites of a [`Quantum`](@ref) Tensor Network.
"""
# noutputs(q::Quantum) = count(!isdual, keys(q.sites))
@deprecate noutputs(tn::AbstractQuantum) nsites(tn; set=:outputs)

"""
    lanes(q::AbstractQuantum)

Returns the lanes of a [`AbstractQuantum`](@ref) Tensor Network.
"""
function lanes(tn::AbstractQuantum)
    return unique(
        Iterators.map(Iterators.flatten([inputs(tn), outputs(tn)])) do site
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

Abstract type representing the socket of a [`Quantum`](@ref) Tensor Network.
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
Base.show(io::IO, q::Quantum) = print(io, "Quantum (inputs=$(ninputs(q)), outputs=$(noutputs(q)))")

Tenet.inds(tn::Quantum, ::Val{:at}, site::Site) = Quantum(tn).sites[site]

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

    tn = conj(TensorNetwork(qtn))

    # rename inner indices
    physical_inds = values(sites)
    virtual_inds = setdiff(inds(tn), physical_inds)
    replace!(tn, map(virtual_inds) do i
        i => Symbol(i, "'")
    end...)

    return Quantum(tn, sites)
end

function sites(tn::AbstractQuantum, ::Val{:set}, query)
    tn = Quantum(tn)

    if query === :all
        collect(keys(tn.sites))
    elseif query === :inputs
        filter(isdual, keys(tn.sites))
    elseif query === :outputs
        filter(!isdual, keys(tn.sites))
    else
        throw(MethodError(sites, (Quantum,), kwargs))
    end
end

# sites(tn::AbstractQuantum, ::Val{:at}, i) = findfirst(i -> i === kwargs[:at], tn.sites)

"""
    nsites(q::Quantum)

Returns the number of sites of a [`Quantum`](@ref) Tensor Network.
"""
function nsites(tn::Quantum; set=:all)
    if set === :all
        length(tn.sites)
    elseif set === :inputs
        length(sites(tn; set))
    elseif set === :outputs
        length(sites(tn; set))
    end
end

"""
    getindex(q::Quantum, site::Site)

Returns the index associated with a site in a [`Quantum`](@ref) Tensor Network.
"""
@deprecate Base.getindex(q::Quantum, site::Site) inds(q; at=site)

# TODO use interfaces/abstract types for better composition of functionality
@inline function Base.replace!(tn::Quantum, old_new::P...) where {P<:Pair}
    return invoke(replace!, Tuple{Quantum,Base.AbstractVecOrTuple{P}}, tn, old_new)
end
@inline Base.replace!(tn::Quantum, old_new::Dict) = replace!(tn, collect(old_new))

function Base.replace!(tn::Quantum, old_new::Base.AbstractVecOrTuple{Pair{Symbol,Symbol}})
    # replace indices in underlying Tensor Network
    replace!(TensorNetwork(tn), old_new)

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
        collect(inputs(b))
    elseif iob === :outputs
        collect(outputs(b))
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

function resetindex!(tn::Quantum; init=1)
    mapping = resetindex!(Val(:return_mapping), TensorNetwork(tn); init)

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
    @assert issetequal(outputs(a), map(adjoint, inputs(b))) "Outputs of $a must match inputs of $b"

    @reindex! outputs(a) => inputs(b)
    tn = merge(TensorNetwork(a), TensorNetwork(b))

    sites = Dict{Site,Symbol}()

    for site in inputs(a)
        sites[site] = inds(a; at=site)
    end

    for site in outputs(b)
        sites[site] = inds(b; at=site)
    end

    return Quantum(tn, sites)
end

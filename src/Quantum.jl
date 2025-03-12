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
    pluggable::PluggableMixin

    # WARN keep them synchronized
    # sites::Dict{Site,Symbol}
    # sitetensors::Dict{Site,Tensor}
end

function Quantum(tn::TensorNetwork, sites::Dict{S,Symbol}=Dict{Site,Symbol}()) where {S<:Site}
    for (_, ind) in sites
        if !hasind(tn, ind)
            error("Index $ind not found in TensorNetwork")
        elseif ind ∉ inds(tn; set=:open)
            error("Index $ind must be open")
        end
    end

    return Quantum(tn, PluggableMixin(sites))
end

Quantum(tn::Quantum) = tn

"""
    TensorNetwork(q::AbstractQuantum)

Return the underlying `TensorNetwork` of an [`AbstractQuantum`](@ref).
"""
TensorNetwork(tn::AbstractQuantum) = Quantum(tn).tn

################################################################################
# TODO refactor this as the one-to-use method on future PR
trait(::TensorNetworkInterface, ::AbstractQuantum) = WrapsTensorNetwork()
unwrap(::TensorNetworkInterface, tn::AbstractQuantum) = TensorNetwork(tn)

trait(::PluggableInterface, ::AbstractQuantum) = WrapsPluggable()
unwrap(::PluggableInterface, tn::AbstractQuantum) = Quantum(tn)

trait(::PluggableInterface, ::Quantum) = WrapsPluggable()
unwrap(::PluggableInterface, tn::Quantum) = tn.pluggable
################################################################################

Base.copy(tn::Quantum) = Quantum(copy(TensorNetwork(tn)), copy(tn.pluggable))

Base.similar(tn::Quantum) = Quantum(similar(TensorNetwork(tn)), copy(tn.pluggable))
Base.zero(tn::Quantum) = Quantum(zero(TensorNetwork(tn)), copy(tn.pluggable))

function Base.:(==)(a::AbstractQuantum, b::AbstractQuantum)
    return Quantum(a).pluggable == Quantum(b).pluggable && ==(TensorNetwork(a), TensorNetwork(b))
end
function Base.isapprox(a::AbstractQuantum, b::AbstractQuantum; kwargs...)
    return Quantum(a).pluggable == Quantum(b).pluggable && isapprox(TensorNetwork(a), TensorNetwork(b); kwargs...)
end

Base.summary(io::IO, tn::AbstractQuantum) = print(io, "$(ntensors(tn))-tensors Quantum")
function Base.show(io::IO, tn::T) where {T<:AbstractQuantum}
    return print(io, "$T (inputs=$(nsites(tn; set=:inputs)), outputs=$(nsites(tn; set=:outputs)))")
end

@deprecate Base.getindex(q::Quantum, site::Site) inds(q; at=site) false

# TODO write a handler for `DeleteEffect{Symbol}` and make `tryprune!` call it
function handle!(tn::AbstractQuantum, effect::DeleteEffect{Tensor})
    tensor = effect.f
    mixin = unwrap(PluggableInterface(), tn)

    targets = inds(tn; set=:physical) ∩ vinds(tensor)
    for target in targets
        site = sites(tn; at=target)
        rmsite!(mixin, site)
    end
end

function handle!(tn::AbstractQuantum, effect::ReplaceEffect{Pair{Symbol,Symbol}})
    handle!(unwrap(PluggableInterface(), tn), effect)
end

#------------------------------------------------------------------------------#

"""
    Base.merge(a::AbstractQuantum, b::AbstractQuantum; reset=true)

Merge multiple [`AbstractQuantum`](@ref) Tensor Networks. If `reset=true`, then all indices are renamed. If `reset=false`, then only the indices of the input/output sites are renamed.

See also: [`merge!`](@ref), [`@align!`](@ref).
"""
Base.merge(tns::AbstractQuantum...; kwargs...) = foldl((a, b) -> merge!(a, b; kwargs...), copy.(tns))
Base.merge!(tns::AbstractQuantum...; kwargs...) = foldl((a, b) -> merge!(a, b; kwargs...), tns)

"""
    Base.merge!(::AbstractQuantum...; reset=true)

Merge in-place multiple [`AbstractQuantum`](@ref) Tensor Networks. If `reset=true`, then all indices are renamed. If `reset=false`, then only the indices of the input/output sites are renamed.

See also: [`merge`](@ref), [`@align!`](@ref).
"""
function Base.merge!(a::AbstractQuantum, b::AbstractQuantum; reset=true)
    @assert adjoint.(sites(b; set=:inputs)) ⊆ sites(a; set=:outputs) "Inputs of b must match outputs of a"
    @assert isdisjoint(setdiff(sites(b; set=:outputs), adjoint.(sites(b; set=:inputs))), sites(a; set=:outputs)) "b cannot create new sites where is not connected"

    @align! outputs(a) => inputs(b) reset = reset
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

    @align! outputs(ψ) => inputs(ϕ) reset = false
    @align! inputs(ψ) => outputs(ϕ) reset = false
    return abs(sqrt(only(contract(merge(TensorNetwork(ψ), TensorNetwork(ϕ)); kwargs...))))
end

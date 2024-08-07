using LinearAlgebra

"""
    Ansatz

[`AbstractQuantum`](@ref) Tensor Network with a predefined structure.

# Notes

  - Any subtype must define `super::Quantum` field or specialize the `Quantum` method.
"""
abstract type Ansatz <: AbstractQuantum end

# TODO maybe we need to change this?
Quantum(@nospecialize tn::Ansatz) = tn.super

Base.:(==)(a::Ansatz, b::Ansatz) = Quantum(a) == Quantum(b)
Base.isapprox(a::Ansatz, b::Ansatz; kwargs...) = isapprox(Quantum(a), Quantum(b); kwargs...)

alias(::A) where {A} = string(A)
function Base.summary(io::IO, tn::A) where {A<:Ansatz}
    return print(io, "$(alias(tn)) (inputs=$(ninputs(tn)), outputs=$(noutputs(tn)))")
end
Base.show(io::IO, tn::A) where {A<:Ansatz} = summary(io, tn)

function Tenet.inds(tn::Ansatz, ::Val{:bond}, (site1, site2)::Tuple{Site,Site}) #, site2::Site)
    @assert site1 ∈ sites(tn) "Site $site1 not found"
    @assert site2 ∈ sites(tn) "Site $site2 not found"
    @assert site1 != site2 "Sites must be different"

    tensor1 = tensors(tn; at=site1)
    tensor2 = tensors(tn; at=site2)

    isdisjoint(inds(tensor1), inds(tensor2)) && return nothing
    return only(inds(tensor1) ∩ inds(tensor2))
end

function Tenet.tensors(tn::Ansatz, ::Val{:between}, (site1, site2)::Tuple{Site,Site})
    @assert site1 ∈ sites(tn) "Site $site1 not found"
    @assert site2 ∈ sites(tn) "Site $site2 not found"
    @assert site1 != site2 "Sites must be different"

    tensor1 = tensors(tn; at=site1)
    tensor2 = tensors(tn; at=site2)

    isdisjoint(inds(tensor1), inds(tensor2)) && return nothing

    return TensorNetwork(tn)[only(inds(tensor1) ∩ inds(tensor2))]
end

struct MissingSchmidtCoefficientsException <: Base.Exception
    bond::NTuple{2,Site}
end

MissingSchmidtCoefficientsException(bond::Vector{<:Site}) = MissingSchmidtCoefficientsException(tuple(bond...))

function Base.showerror(io::IO, e::MissingSchmidtCoefficientsException)
    return print(io, "Can't access the spectrum on bond $(e.bond)")
end

function LinearAlgebra.norm(ψ::Ansatz, p::Real=2; kwargs...)
    p == 2 || throw(ArgumentError("only L2-norm is implemented yet"))

    return LinearAlgebra.norm2(ψ; kwargs...)
end

function LinearAlgebra.norm2(ψ::Ansatz; kwargs...)
    return abs(sqrt(only(contract(merge(TensorNetwork(ψ), TensorNetwork(ψ')); kwargs...))))
end

# Traits
abstract type Boundary end
struct Open <: Boundary end
struct Periodic <: Boundary end

function boundary end

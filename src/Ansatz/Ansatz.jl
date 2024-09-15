using KeywordDispatch
using LinearAlgebra
using Graphs
using MetaGraphsNext

abstract type AbstractAnsatz <: AbstractQuantum end

"""
    Ansatz

[`AbstractQuantum`](@ref) Tensor Network with a preserving structure.
"""
struct Ansatz <: AbstractAnsatz
    tn::Quantum
    lattice::MetaGraph

    function Ansatz(tn, lattice)
        if !issetequal(site(tn), labels(lattice))
            throw(ArgumentError("Sites of the tensor network and the lattice must be equal"))
        end
        return new(tn, lattice)
    end
end

Ansatz(tn::Ansatz) = tn
Quantum(tn::AbstractAnsatz) = Ansatz(tn).tn
lattice(tn::AbstractAnsatz) = Ansatz(tn).lattice

function Base.isapprox(a::AbstractAnsatz, b::AbstractAnsatz; kwargs...)
    return ==(latice.((a, b))...) && isapprox(Quantum(a), Quantum(b); kwargs...)
end

function neighbors(tn::AbstractAnsatz, site::Site)
    # TODO
    # return neighbors(lattice(tn), site)
end

function isneighbor(tn::AbstractAnsatz, a::Site, b::Site)
    # TODO
    # return isneighbor(lattice(tn), a, b)
end

@kwmethod function inds(tn::AbstractAnsatz; bond)
    (site1, site2) = bond
    @assert site1 ∈ sites(tn) "Site $site1 not found"
    @assert site2 ∈ sites(tn) "Site $site2 not found"
    @assert site1 != site2 "Sites must be different"
    @assert isneighbor(tn, site1, site2) "Sites must be neighbors"

    tensor1 = tensors(tn; at=site1)
    tensor2 = tensors(tn; at=site2)

    isdisjoint(inds(tensor1), inds(tensor2)) && return nothing
    return only(inds(tensor1) ∩ inds(tensor2))
end

@kwmethod function Tenet.tensors(tn::AbstractAnsatz; between)
    (site1, site2) = between
    @assert site1 ∈ sites(tn) "Site $site1 not found"
    @assert site2 ∈ sites(tn) "Site $site2 not found"
    @assert site1 != site2 "Sites must be different"
    @assert isneighbor(tn, site1, site2) "Sites must be neighbors"

    tensor1 = tensors(tn; at=site1)
    tensor2 = tensors(tn; at=site2)

    isdisjoint(inds(tensor1), inds(tensor2)) && return nothing

    return tn[only(inds(tensor1) ∩ inds(tensor2))]
end

struct MissingSchmidtCoefficientsException <: Base.Exception
    bond::NTuple{2,Site}
end

MissingSchmidtCoefficientsException(bond::Vector{<:Site}) = MissingSchmidtCoefficientsException(tuple(bond...))

function Base.showerror(io::IO, e::MissingSchmidtCoefficientsException)
    return print(io, "Can't access the spectrum on bond $(e.bond)")
end

# Traits
abstract type Boundary end
struct Open <: Boundary end
struct Periodic <: Boundary end

function boundary end

abstract type Form end
struct NonCanonical <: Form end
struct MixedCanonical <: Form
    orthogonality_center::Union{Site,Vector{Site}}
end
struct Canonical <: Form end

function form end

using KeywordDispatch
using LinearAlgebra
using Graphs
using MetaGraphsNext

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

struct MissingSchmidtCoefficientsException <: Base.Exception
    bond::NTuple{2,Site}
end

MissingSchmidtCoefficientsException(bond::Vector{<:Site}) = MissingSchmidtCoefficientsException(tuple(bond...))

function Base.showerror(io::IO, e::MissingSchmidtCoefficientsException)
    return print(io, "Can't access the spectrum on bond $(e.bond)")
end

abstract type AbstractAnsatz <: AbstractQuantum end

"""
    Ansatz

[`AbstractQuantum`](@ref) Tensor Network with a preserving structure.
"""
struct Ansatz <: AbstractAnsatz
    tn::Quantum
    lattice::MetaGraph

    function Ansatz(tn, lattice)
        if !issetequal(lanes(tn), labels(lattice))
            throw(ArgumentError("Sites of the tensor network and the lattice must be equal"))
        end
        return new(tn, lattice)
    end
end

Ansatz(tn::Ansatz) = tn
Quantum(tn::AbstractAnsatz) = Ansatz(tn).tn

Base.copy(tn::Ansatz) = Ansatz(copy(Quantum(tn)), copy(lattice(tn)))
Base.similar(tn::Ansatz) = Ansatz(similar(Quantum(tn)), copy(lattice(tn)))
Base.zero(tn::Ansatz) = Ansatz(zero(Quantum(tn)), copy(lattice(tn)))

lattice(tn::AbstractAnsatz) = Ansatz(tn).lattice

function Base.isapprox(a::AbstractAnsatz, b::AbstractAnsatz; kwargs...)
    return ==(latice.((a, b))...) && isapprox(Quantum(a), Quantum(b); kwargs...)
end

function Graphs.neighbors(tn::AbstractAnsatz, site::Site)
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

@kwmethod tensors(tn::AbstractAnsatz; bond) = tn[inds(tn; bond)]
@kwmethod function tensors(tn::AbstractAnsatz; between)
    Base.depwarn(
        "`tensors(tn; between)` is deprecated, use `tensors(tn; bond)` instead.",
        ((Base.Core).Typeof(tensors)).name.mt.name,
    )
    return tensors(tn; bond=between)
end

@kwmethod contract!(tn::AbstractAnsatz; bond) = contract!(tn, inds(tn; bond))

canonize(tn::AbstractAnsatz, args...; kwargs...) = canonize!(deepcopy(tn), args...; kwargs...)
canonize_site(tn::AbstractAnsatz, args...; kwargs...) = canonize_site!(deepcopy(tn), args...; kwargs...)

"""
    truncate(tn::AbstractAnsatz, bond; threshold = nothing, maxdim = nothing)

Like [`truncate!`](@ref), but returns a new tensor network instead of modifying the original one.
"""
truncate(tn::AbstractAnsatz, args...; kwargs...) = truncate!(deepcopy(tn), args...; kwargs...)

"""
    truncate!(tn::AbstractAnsatz, bond; threshold = nothing, maxdim = nothing)

Truncate the dimension of the virtual `bond`` of an [`Ansatz`](@ref) Tensor Network by keeping only the `maxdim` largest Schmidt coefficients or those larger than`threshold`.

# Notes

  - Either `threshold` or `maxdim` must be provided. If both are provided, `maxdim` is used.
  - The bond must contain the Schmidt coefficients, i.e. a site canonization must be performed before calling `truncate!`.
"""
function truncate!(tn::AbstractAnsatz, bond; threshold=nothing, maxdim=nothing)
    @assert isnothing(maxdim) ⊻ isnothing(threshold) "Either `threshold` or `maxdim` must be provided"

    spectrum = parent(tensors(tn; bond))
    vind = inds(tn; bond)

    maxdim = isnothing(maxdim) ? size(tn, vind) : maxdim
    threshold = isnothing(threshold) ? 1e-16 : threshold

    extent = findfirst(1:maxdim) do i
        abs(spectrum[i]) < threshold
    end

    slice!(tn, vind, extent)

    return tn
end

function expect(ψ::AbstractAnsatz, observables; bra=copy(ψ))
    ϕ = bra

    # TODO is this ok?
    for observable in observables
        evolve!(ϕ, observable)
    end

    return overlap(ϕ, ψ)
end

overlap(a::AbstractAnsatz, b::AbstractAnsatz) = contract(merge(a, copy(b)'))

function evolve!(ψ::AbstractAnsatz, gate; threshold=nothing, maxdim=nothing, renormalize=false)
    return simple_update!(ψ, gate; threshold, maxdim, renormalize)
end

# by popular demand (Stefano, I'm looking at you), I aliased `apply!` to `evolve!`
const apply! = evolve!

function simple_update!(ψ::AbstractAnsatz, gate; threshold=nothing, maxdim=nothing, kwargs...)
    @assert issetequal(adjoint.(sites(gate; set=:inputs)), sites(gate; set=:outputs)) "Inputs of the gate must match outputs"
    @assert isneighbor(ψ, lanes(gate)...) "Gate must act on neighboring sites"

    if nlanes(gate) == 1
        return simple_update_1site!(ψ, gate)
    end

    return simple_update!(form(ψ), ψ, gate; kwargs...)
end

function simple_update_1site!(ψ::AbstractAnsatz, gate)
    @assert nlanes(gate) == 1 "Gate must act only on one lane"
    @assert ninputs(gate) == 1 "Gate must have only one input"
    @assert noutputs(gate) == 1 "Gate must have only one output"

    # shallow copy to avoid problems if errors in mid execution
    gate = copy(gate)

    contracting_index = gensym(:tmp)
    targetsite = only(sites(gate; set=:inputs))'

    # reindex contracting index
    replace!(ψ, inds(ψ; at=targetsite) => contracting_index)
    replace!(gate, inds(gate; at=targetsite') => contracting_index)

    # reindex output of gate to match TN sitemap
    replace!(gate, inds(gate; at=only(sites(gate; set=:outputs))) => inds(ψ; at=targetsite))

    # contract gate with TN
    merge!(ψ, gate)
    return contract!(ψ, contracting_index)
end

function simple_update!(::NonCanonical, ψ::AbstractAnsatz, gate; threshold=nothing, maxdim=nothing)
    @assert nlanes(gate) == 2 "Only 2-site gates are supported currently"

    # shallow copy to avoid problems if errors in mid execution
    gate = copy(gate)

    merge!(ψ, gate)
    contract!(ψ, inds(gate; set=:inputs))

    # TODO split

    return ψ
end

# TODO move non-canonical code to method above
# TODO remove `renormalize` argument
# TODO refactor code
function simple_update!(::Canonical, ψ::AbstractAnsatz, gate; threshold, maxdim, renormalize=false, iscanonical=false)
    # shallow copy to avoid problems if errors in mid execution
    gate = copy(gate)

    bond = sitel, siter = minmax(sites(gate; set=:outputs)...)
    left_inds::Vector{Symbol} = !isnothing(leftindex(ψ, sitel)) ? [leftindex(ψ, sitel)] : Symbol[]
    right_inds::Vector{Symbol} = !isnothing(rightindex(ψ, siter)) ? [rightindex(ψ, siter)] : Symbol[]

    virtualind::Symbol = inds(ψ; bond=bond)

    iscanonical ? contract_2sitewf!(ψ, bond) : contract!(TensorNetwork(ψ), virtualind)

    # reindex contracting index
    contracting_inds = [gensym(:tmp) for _ in sites(gate; set=:inputs)]
    replace!(
        ψ,
        map(zip(sites(gate; set=:inputs), contracting_inds)) do (site, contracting_index)
            inds(ψ; at=site') => contracting_index
        end,
    )
    replace!(
        gate,
        map(zip(sites(gate; set=:inputs), contracting_inds)) do (site, contracting_index)
            inds(gate; at=site) => contracting_index
        end,
    )

    # replace output indices of the gate for gensym indices
    output_inds = [gensym(:out) for _ in sites(gate; set=:outputs)]
    replace!(
        gate,
        map(zip(sites(gate; set=:outputs), output_inds)) do (site, out)
            inds(gate; at=site) => out
        end,
    )

    # reindex output of gate to match TN sitemap
    for site in sites(gate; set=:outputs)
        if inds(ψ; at=site) != inds(gate; at=site)
            replace!(gate, inds(gate; at=site) => inds(ψ; at=site))
        end
    end

    # contract physical inds
    merge!(ψ, gate)
    contract!(ψ, contracting_inds)

    # decompose using SVD
    push!(left_inds, inds(ψ; at=sitel))
    push!(right_inds, inds(ψ; at=siter))

    if iscanonical
        unpack_2sitewf!(ψ, bond, left_inds, right_inds, virtualind)
    else
        svd!(ψ; left_inds, right_inds, virtualind)
    end
    # truncate virtual index
    if any(!isnothing, [threshold, maxdim])
        truncate!(ψ, bond; threshold, maxdim)

        # renormalize the bond
        if renormalize && iscanonical
            λ = tensors(ψ; between=bond)
            replace!(ψ, λ => normalize(λ)) # TODO this can be replaced by `normalize!(λ)`
        elseif renormalize && !iscanonical
            normalize!(ψ, bond[1])
        end
    end

    return ψ
end

# TODO refactor code
"""
    contract_2sitewf!(ψ::AbstractAnsatz, bond)

For a given [`AbstractAnsatz`](@ref) in the canonical form, creates the two-site wave function θ with Λᵢ₋₁Γᵢ₋₁ΛᵢΓᵢΛᵢ₊₁,
where i is the `bond`, and replaces the Γᵢ₋₁ΛᵢΓᵢ tensors with θ.
"""
function contract_2sitewf!(ψ::AbstractAnsatz, bond)
    @assert form(ψ) == Canonical() "The tensor network must be in canonical form"

    sitel, siter = bond # TODO Check if bond is valid
    (0 < id(sitel) < nsites(ψ) || 0 < id(siter) < nsites(ψ)) ||
        throw(ArgumentError("The sites in the bond must be between 1 and $(nsites(ψ))"))

    Λᵢ₋₁ = id(sitel) == 1 ? nothing : tensors(ψ; between=(Site(id(sitel) - 1), sitel))
    Λᵢ₊₁ = id(sitel) == nsites(ψ) - 1 ? nothing : tensors(ψ; between=(siter, Site(id(siter) + 1)))

    !isnothing(Λᵢ₋₁) && contract!(ψ; between=(Site(id(sitel) - 1), sitel), direction=:right, delete_Λ=false)
    !isnothing(Λᵢ₊₁) && contract!(ψ; between=(siter, Site(id(siter) + 1)), direction=:left, delete_Λ=false)

    contract!(ψ, inds(ψ; bond=bond))

    return ψ
end

# TODO refactor code
"""
    unpack_2sitewf!(ψ::AbstractAnsatz, bond)

For a given [`AbstractAnsatz`](@ref) that contains a two-site wave function θ in a bond, it decomposes θ into the canonical
form: Γᵢ₋₁ΛᵢΓᵢ, where i is the `bond`.
"""
function unpack_2sitewf!(ψ::AbstractAnsatz, bond, left_inds, right_inds, virtualind)
    @assert form(ψ) == Canonical() "The tensor network must be in canonical form"

    sitel, siter = bond # TODO Check if bond is valid
    (0 < id(sitel) < nsites(ψ) || 0 < id(site_r) < nsites(ψ)) ||
        throw(ArgumentError("The sites in the bond must be between 1 and $(nsites(ψ))"))

    Λᵢ₋₁ = id(sitel) == 1 ? nothing : tensors(ψ; between=(Site(id(sitel) - 1), sitel))
    Λᵢ₊₁ = id(siter) == nsites(ψ) ? nothing : tensors(ψ; between=(siter, Site(id(siter) + 1)))

    # do svd of the θ tensor
    θ = tensors(ψ; at=sitel)
    U, s, Vt = svd(θ; left_inds, right_inds, virtualind)

    # contract with the inverse of Λᵢ and Λᵢ₊₂
    Γᵢ₋₁ =
        isnothing(Λᵢ₋₁) ? U : contract(U, Tensor(diag(pinv(Diagonal(parent(Λᵢ₋₁)); atol=1e-32)), inds(Λᵢ₋₁)); dims=())
    Γᵢ =
        isnothing(Λᵢ₊₁) ? Vt : contract(Tensor(diag(pinv(Diagonal(parent(Λᵢ₊₁)); atol=1e-32)), inds(Λᵢ₊₁)), Vt; dims=())

    delete!(ψ, θ)

    push!(ψ, Γᵢ₋₁)
    push!(ψ, s)
    push!(ψ, Γᵢ)

    return ψ
end

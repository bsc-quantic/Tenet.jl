using KeywordDispatch
using LinearAlgebra
using Graphs: Graphs

# Traits
"""
    Boundary

Abstract type representing the boundary condition trait of a [`AbstractAnsatz`](@ref) Tensor Network.
"""
abstract type Boundary end

"""
    Open

[`Boundary`](@ref) trait representing an open boundary condition.
"""
struct Open <: Boundary end

"""
    Periodic

[`Boundary`](@ref) trait representing a periodic boundary condition.
"""
struct Periodic <: Boundary end

function boundary end

"""
    Form

Abstract type representing the canonical form trait of a [`AbstractAnsatz`](@ref) Tensor Network.
"""
abstract type Form end

Base.copy(x::Form) = x

"""
    NonCanonical

[`Form`](@ref) trait representing a [`AbstractAnsatz`](@ref) Tensor Network in non-canonical form.
"""
struct NonCanonical <: Form end

"""
    MixedCanonical

[`Form`](@ref) trait representing a [`AbstractAnsatz`](@ref) Tensor Network in mixed-canonical form.

  - The orthogonality center is a [`Site`](@ref) or a vector of [`Site`](@ref)s. The tensors to the
    left of the orthogonality center are left-canonical and the tensors to the right are right-canonical.
"""
struct MixedCanonical <: Form
    orthog_center::Union{Site,Vector{<:Site}}
end

Base.copy(x::MixedCanonical) = MixedCanonical(copy(x.orthog_center))

"""
    Canonical

[`Form`](@ref) trait representing a [`AbstractAnsatz`](@ref) Tensor Network in canonical form or Vidal gauge.
"""
struct Canonical <: Form end

"""
    form(tn::AbstractAnsatz)

Return the canonical form of the [`AbstractAnsatz`](@ref) Tensor Network.
"""
function form end

struct MissingSchmidtCoefficientsException <: Base.Exception
    bond::NTuple{2,Site}
end

MissingSchmidtCoefficientsException(bond::Vector{<:Site}) = MissingSchmidtCoefficientsException(tuple(bond...))

function Base.showerror(io::IO, e::MissingSchmidtCoefficientsException)
    return print(io, "Can't access the spectrum on bond $(e.bond)")
end

"""
    AbstractAnsatz

Abstract type for [`Ansatz`](@ref)-derived types.
Its subtypes must implement conversion or extraction to the underlying [`Ansatz`](@ref).
"""
abstract type AbstractAnsatz <: AbstractQuantum end

"""
    Ansatz

[`AbstractQuantum`](@ref) Tensor Network together with a [`Lattice`](@ref) for connectivity information between [`Site`](@ref)s.
"""
struct Ansatz <: AbstractAnsatz
    tn::Quantum
    lattice::Lattice

    function Ansatz(tn, lattice)
        if !issetequal(lanes(tn), Graphs.vertices(lattice))
            throw(ArgumentError("Sites of the tensor network and the lattice must be equal"))
        end
        return new(tn, lattice)
    end
end

Ansatz(tn::Ansatz) = tn

"""
    Quantum(tn::AbstractAnsatz)

Return the underlying [`Quantum`](@ref) Tensor Network of an [`AbstractAnsatz`](@ref).
"""
Quantum(tn::AbstractAnsatz) = Ansatz(tn).tn

# default form
form(::AbstractAnsatz) = NonCanonical()

Base.copy(tn::Ansatz) = Ansatz(copy(Quantum(tn)), copy(lattice(tn)))
Base.similar(tn::Ansatz) = Ansatz(similar(Quantum(tn)), copy(lattice(tn)))
Base.zero(tn::Ansatz) = Ansatz(zero(Quantum(tn)), copy(lattice(tn)))

"""
    lattice(tn::AbstractAnsatz)

Return the [`Lattice`](@ref) of the [`AbstractAnsatz`](@ref) Tensor Network.
"""
lattice(tn::AbstractAnsatz) = Ansatz(tn).lattice

function Base.isapprox(a::AbstractAnsatz, b::AbstractAnsatz; kwargs...)
    return ==(latice.((a, b))...) && isapprox(Quantum(a), Quantum(b); kwargs...)
end

"""
    neighbors(tn::AbstractAnsatz, site::Site)

Return the neighboring sites of a given [`Site`](@ref) in the [`Lattice`](@ref) of the [`AbstractAnsatz`](@ref) Tensor Network.
"""
Graphs.neighbors(tn::AbstractAnsatz, site::Site) = Graphs.neighbors(lattice(tn), site)

"""
    has_edge(tn::AbstractAnsatz, a::Site, b::Site)

Check whether there is an edge between two [`Site`](@ref)s in the [`Lattice`](@ref) of the [`AbstractAnsatz`](@ref) Tensor Network.
"""
Graphs.has_edge(tn::AbstractAnsatz, a::Site, b::Site) = Graphs.has_edge(lattice(tn), a, b)

"""
    inds(tn::AbstractAnsatz; bond)

Return the index of the virtual bond between two [`Site`](@ref)s in a [`AbstractAnsatz`](@ref) Tensor Network.
"""
@kwmethod function inds(tn::AbstractAnsatz; bond)
    (site1, site2) = bond
    @assert site1 ∈ sites(tn) "Site $site1 not found"
    @assert site2 ∈ sites(tn) "Site $site2 not found"
    @assert site1 != site2 "Sites must be different"
    @assert Graphs.has_edge(tn, site1, site2) "Sites must be neighbors"

    tensor1 = tensors(tn; at=site1)
    tensor2 = tensors(tn; at=site2)

    isdisjoint(inds(tensor1), inds(tensor2)) && return nothing
    return only(inds(tensor1) ∩ inds(tensor2))
end

"""
    tensors(tn::AbstractAnsatz; bond)

Return the [`Tensor`](@ref) in a virtual bond between two [`Site`](@ref)s in a [`AbstractAnsatz`](@ref) Tensor Network.

# Notes

  - If the [`AbstractAnsatz`](@ref) Tensor Network is in the canonical form, Tenet stores the Schmidt coefficients of the bond in a vector connected to the bond hyperedge between the two sites and the vector.
  - If the bond contains no Schmidt coefficients, this method will throw a `MissingSchmidtCoefficientsException`.
"""
@kwmethod function tensors(tn::AbstractAnsatz; bond)
    vind = inds(tn; bond)
    tensor = tensors(tn, [vind]) do vinds, indices
        indices == vinds
    end
    isempty(tensor) && throw(MissingSchmidtCoefficientsException(bond))
    return only(tensor)
end

@kwmethod function tensors(tn::AbstractAnsatz; between)
    Base.depwarn(
        "`tensors(tn; between)` is deprecated, use `tensors(tn; bond)` instead.",
        ((Base.Core).Typeof(tensors)).name.mt.name,
    )
    return tensors(tn; bond=between)
end

"""
    contract!(tn::AbstractAnsatz; bond)

Contract the virtual bond between two [`Site`](@ref)s in a [`AbstractAnsatz`](@ref) Tensor Network.
"""
@kwmethod contract!(tn::AbstractAnsatz; bond) = contract!(tn, inds(tn; bond))

"""
    canonize!(tn::AbstractAnsatz)

Transform an [`AbstractAnsatz`](@ref) Tensor Network into the canonical form (aka Vidal gauge); i.e. the singular values matrix Λᵢ between each tensor Γᵢ₋₁ and Γᵢ.
"""
function canonize! end

"""
    canonize(tn::AbstractAnsatz)

Like [`canonize!`](@ref), but returns a new Tensor Network instead of modifying the original one.
"""
canonize(tn::AbstractAnsatz, args...; kwargs...) = canonize!(deepcopy(tn), args...; kwargs...)

"""
    mixed_canonize!(tn::AbstractAnsatz, orthog_center)

Transform an [`AbstractAnsatz`](@ref) Tensor Network into the mixed-canonical form, that is,
for `i < orthog_center` the tensors are left-canonical and for `i >= orthog_center` the tensors are right-canonical,
and in the `orthog_center` there is a tensor with the Schmidt coefficients in it.
"""
function mixed_canonize! end

"""
    mixed_canonize(tn::AbstractAnsatz, orthog_center)

Like [`mixed_canonize!`](@ref), but returns a new Tensor Network instead of modifying the original one.
"""
mixed_canonize(tn::AbstractAnsatz, args...; kwargs...) = mixed_canonize!(deepcopy(tn), args...; kwargs...)

canonize_site(tn::AbstractAnsatz, args...; kwargs...) = canonize_site!(deepcopy(tn), args...; kwargs...)

"""
    normalize!(ψ::AbstractAnsatz, at)

Normalize the state at a given [`Site`](@ref) or bond in a [`AbstractAnsatz`](@ref) Tensor Network.
"""
LinearAlgebra.normalize(ψ::AbstractAnsatz, site) = normalize!(copy(ψ), site)

"""
    isisometry(tn::AbstractAnsatz, site; dir, kwargs...)

Check if the tensor at a given [`Site`](@ref) in a [`AbstractAnsatz`](@ref) Tensor Network is an isometry.
The `dir` keyword argument specifies the direction of the isometry to check.
"""
function isisometry end

"""
    truncate(tn::AbstractAnsatz, bond; threshold = nothing, maxdim = nothing)

Like [`truncate!`](@ref), but returns a new Tensor Network instead of modifying the original one.
"""
Base.truncate(tn::AbstractAnsatz, args...; kwargs...) = truncate!(deepcopy(tn), args...; kwargs...)

"""
    truncate!(tn::AbstractAnsatz, bond; threshold = nothing, maxdim = nothing)

Truncate the dimension of the virtual `bond`` of an [`Ansatz`](@ref) Tensor Network. Dispatches to the appropriate method based on the [`form`](@ref) of the Tensor Network:

  - If the Tensor Network is in the [`MixedCanonical`](@ref) form, the bond is truncated by moving the orthogonality center to the bond and keeping the `maxdim` largest **Schmidt coefficients** or those larger than `threshold`.
  - If the Tensor Network is in the [`Canonical`](@ref) form, the bond is truncated by keeping the `maxdim` largest **Schmidt coefficients** or those larger than `threshold`, and then recanonizing the Tensor Network.
  - If the Tensor Network is in the [`NonCanonical`](@ref) form, the bond is truncated by contracting the bond, performing an SVD and keeping the `maxdim` largest **singular values** or those larger than `threshold`.

# Notes

  - Either `threshold` or `maxdim` must be provided. If both are provided, `maxdim` is used.
"""
function truncate!(tn::AbstractAnsatz, bond; threshold=nothing, maxdim=nothing, kwargs...)
    all(isnothing, (threshold, maxdim)) && return tn

    return truncate!(form(tn), tn, bond; threshold, maxdim, kwargs...)
end

"""
    truncate!(::NonCanonical, tn::AbstractAnsatz, bond; threshold, maxdim, compute_local_svd=true)

Truncate the dimension of the virtual `bond` of a [`NonCanonical`](@ref) Tensor Network by contracting the bond, performing an SVD and keeping the `maxdim` largest **singular values** or those larger than `threshold`.

# Arguments

  - `tn`: The [`AbstractAnsatz`](@ref) Tensor Network.
  - `bond`: The bond to truncate.

# Keyword Arguments

  - `threshold`: The threshold to truncate the bond dimension.
  - `maxdim`: The maximum bond dimension to keep.
  - `compute_local_svd`: Whether to compute the local SVD of the bond. If `true`, it will contract the bond and perform a SVD to get the local singular values. Defaults to `true`.
  - `normalize`: Whether to normalize the state at the bond after truncation. Defaults to `false`.
"""
function truncate!(::NonCanonical, tn::AbstractAnsatz, bond; threshold, maxdim, compute_local_svd=true, normalize=false)
    virtualind = inds(tn; bond)

    if compute_local_svd
        tₗ = tensors(tn; at=min(bond...))
        tᵣ = tensors(tn; at=max(bond...))
        contract!(tn; bond)

        left_inds = filter(!=(virtualind), inds(tₗ))
        right_inds = filter(!=(virtualind), inds(tᵣ))
        svd!(tn; left_inds, right_inds, virtualind=virtualind)
    end

    spectrum = parent(tensors(tn; bond))

    maxdim = isnothing(maxdim) ? size(tn, virtualind) : min(maxdim, length(spectrum))

    extent = if isnothing(threshold)
        1:maxdim
    else
        # Find the first index where the condition is met
        found_index = findfirst(1:maxdim) do i
            abs(spectrum[i]) < threshold
        end

        # If no index is found, return 1:length(spectrum), otherwise calculate the range
        1:(isnothing(found_index) ? maxdim : found_index - 1)
    end

    slice!(tn, virtualind, extent)
    sliced_bond = tensors(tn; bond)

    # Note: Inplace normalization of the inner arrays may be more efficient
    normalize && replace!(tn, sliced_bond => sliced_bond ./ norm(tn))

    return tn
end

function truncate!(::MixedCanonical, tn::AbstractAnsatz, bond; kwargs...)
    # move orthogonality center to bond
    mixed_canonize!(tn, bond)

    return truncate!(NonCanonical(), tn, bond; compute_local_svd=true, kwargs...)
end

"""
    truncate!(::Canonical, tn::AbstractAnsatz, bond; canonize=true, kwargs...)

Truncate the dimension of the virtual `bond` of a [`Canonical`](@ref) Tensor Network by keeping the `maxdim` largest
**Schmidt coefficients** or those larger than `threshold`, and then canonizes the Tensor Network if `canonize` is `true`.
"""
function truncate!(::Canonical, tn::AbstractAnsatz, bond; canonize=true, kwargs...)
    truncate!(NonCanonical(), tn, bond; compute_local_svd=false, kwargs...)

    canonize && canonize!(tn)

    return tn
end

"""
    overlap(a::AbstractAnsatz, b::AbstractAnsatz)

Compute the overlap between two [`AbstractAnsatz`](@ref) Tensor Networks.
"""
overlap(a::AbstractAnsatz, b::AbstractAnsatz) = contract(merge(a, copy(b)'))

"""
    expect(ψ::AbstractAnsatz, observable)

Compute the expectation value of an observable on a [`AbstractAnsatz`](@ref) Tensor Network.

# Arguments

  - `ψ`: Tensor Network representing the state.
  - `observable`: The observable to compute the expectation value. If a `Vector` or `Tuple` of observables is provided, the sum of the expectation values is returned.

# Keyword Arguments

  - `bra`: The bra state. Defaults to a copy of `ψ`.
"""
expect(ψ::AbstractAnsatz, observable; bra=copy(ψ)) = contract(merge(ψ, observable, bra'))

function expect(ψ::AbstractAnsatz, observables::AbstractVecOrTuple; bra=copy(ψ))
    sum(observables) do observable
        expect(ψ, observable; bra=copy(bra))
    end
end

"""
    evolve!(ψ::AbstractAnsatz, gate; threshold = nothing, maxdim = nothing, normalize = false)

Evolve (through time) a [`AbstractAnsatz`](@ref) Tensor Network with a `gate` operator.

!!! note

    Currently only the "Simple Update" algorithm is implemented.

# Arguments

  - `ψ`: Tensor Network representing the state.
  - `gate`: The gate operator to evolve the state with.

# Keyword Arguments

  - `threshold`: The threshold to truncate the bond dimension.
  - `maxdim`: The maximum bond dimension to keep.
  - `normalize`: Whether to normalize the state after truncation.

# Notes

  - The gate must act on neighboring sites according to the [`Lattice`](@ref) of the Tensor Network.
  - The gate must have the same number of inputs and outputs.
  - Currently only the "Simple Update" algorithm is used and the gate must be a 1-site or 2-site operator.
"""
function evolve!(ψ::AbstractAnsatz, gate; threshold=nothing, maxdim=nothing, normalize=false, kwargs...)
    return simple_update!(ψ, gate; threshold, maxdim, normalize, kwargs...)
end

# by popular demand (Stefano, I'm looking at you), I aliased `apply!` to `evolve!`
const apply! = evolve!

"""
    simple_update!(ψ::AbstractAnsatz, gate; threshold = nothing, maxdim = nothing, kwargs...)

Update a [`AbstractAnsatz`](@ref) Tensor Network with a `gate` operator using the "Simple Update" algorithm.
`kwargs` are passed to the `truncate!` method in the case of a multi-site gate.

!!! warning

    Currently only 1-site and 2-site gates are supported.

# Arguments

  - `ψ`: Tensor Network representing the state.
  - `gate`: The gate operator to update the state with.

# Keyword Arguments

  - `threshold`: The threshold to truncate the bond dimension.
  - `maxdim`: The maximum bond dimension to keep.
  - `normalize`: Whether to normalize the state after truncation.

# Notes

  - If both `threshold` and `maxdim` are provided, `maxdim` is used.
"""
function simple_update!(ψ::AbstractAnsatz, gate; threshold=nothing, maxdim=nothing, kwargs...)
    @assert issetequal(adjoint.(sites(gate; set=:inputs)), sites(gate; set=:outputs)) "Inputs of the gate must match outputs"

    if nlanes(gate) == 1
        return simple_update_1site!(ψ, gate)
    elseif nlanes(gate) == 2
        return simple_update_2site!(form(ψ), ψ, gate; threshold, maxdim, kwargs...)
    else
        throw(ArgumentError("Only 1-site and 2-site gates are currently supported"))
    end
end

# TODO a lot of problems with merging... maybe we shouldn't merge manually
function simple_update_1site!(ψ::AbstractAnsatz, gate)
    @assert nlanes(gate) == 1 "Gate must act only on one lane"
    @assert ninputs(gate) == 1 "Gate must have only one input"
    @assert noutputs(gate) == 1 "Gate must have only one output"

    # shallow copy to avoid problems if errors in mid execution
    gate = copy(gate)
    resetindex!(gate; init=ninds(ψ))

    contracting_index = gensym(:tmp)
    targetsite = only(sites(gate; set=:inputs))'

    # reindex output of gate to match TN sitemap
    replace!(gate, inds(gate; at=only(sites(gate; set=:outputs))) => inds(ψ; at=targetsite))

    # reindex contracting index
    replace!(ψ, inds(ψ; at=targetsite) => contracting_index)
    replace!(gate, inds(gate; at=targetsite') => contracting_index)

    # contract gate with TN
    merge!(ψ, gate; reset=false)
    return contract!(ψ, contracting_index)
end

function simple_update_2site!(::MixedCanonical, ψ::AbstractAnsatz, gate; kwargs...)
    return simple_update_2site!(NonCanonical(), ψ, gate; kwargs...)
end

function simple_update_2site!(::NonCanonical, ψ::AbstractAnsatz, gate; kwargs...)
    @assert Graphs.has_edge(ψ, lanes(gate)...) "Gate must act on neighboring sites"

    # shallow copy to avoid problems if errors in mid execution
    gate = copy(gate)

    # contract involved sites
    bond = (sitel, siter) = extrema(lanes(gate))
    vind = inds(ψ; bond)
    linds = filter(!=(vind), inds(tensors(ψ; at=sitel)))
    rinds = filter(!=(vind), inds(tensors(ψ; at=siter)))
    contract!(ψ; bond)

    # TODO replace for `merge!` when #243 is fixed
    # reindex contracting indices to temporary names to avoid issues
    oinds = Dict(site => inds(ψ; at=site) for site in sites(gate; set=:outputs))
    tmpinds = Dict(site => gensym(:tmp) for site in sites(gate; set=:inputs))
    replace!(gate, [inds(gate; at=site) => i for (site, i) in tmpinds])
    replace!(ψ, [inds(ψ; at=site') => i for (site, i) in tmpinds])

    # NOTE `replace!` is getting confused when a index is already there even if it would be overriden
    # TODO fix this to be handled in one call -> replace when #244 is fixed
    replace!(gate, [inds(gate; at=site) => gensym() for (site, i) in oinds])
    replace!(gate, [inds(gate; at=site) => i for (site, i) in oinds])

    # contract physical inds with gate
    merge!(ψ, gate; reset=false)
    contract!(ψ, inds(gate; set=:inputs))

    # decompose using SVD
    svd!(ψ; left_inds=linds, right_inds=rinds, virtualind=vind)

    # truncate virtual index
    truncate!(ψ, collect(bond); kwargs...)

    return ψ
end

# TODO remove `normalize` argument?
function simple_update_2site!(::Canonical, ψ::AbstractAnsatz, gate; threshold, maxdim, normalize=false, canonize=true)
    # Contract the exterior Λ tensors
    sitel, siter = extrema(lanes(gate))
    (0 < id(sitel) < nsites(ψ) || 0 < id(siter) < nsites(ψ)) ||
        throw(ArgumentError("The sites in the bond must be between 1 and $(nsites(ψ))"))

    Λᵢ₋₁ = id(sitel) == 1 ? nothing : tensors(ψ; between=(Site(id(sitel) - 1), sitel))
    Λᵢ₊₁ = id(sitel) == nsites(ψ) - 1 ? nothing : tensors(ψ; between=(siter, Site(id(siter) + 1)))

    !isnothing(Λᵢ₋₁) && contract!(ψ; between=(Site(id(sitel) - 1), sitel), direction=:right, delete_Λ=false)
    !isnothing(Λᵢ₊₁) && contract!(ψ; between=(siter, Site(id(siter) + 1)), direction=:left, delete_Λ=false)

    simple_update_2site!(NonCanonical(), ψ, gate; threshold, maxdim, normalize=false, canonize=false)

    # contract the updated tensors with the inverse of Λᵢ and Λᵢ₊₂, to get the new Γ tensors
    U, Vt = tensors(ψ; at=sitel), tensors(ψ; at=siter)
    Γᵢ₋₁ = if isnothing(Λᵢ₋₁)
        U
    else
        contract(U, Tensor(diag(pinv(Diagonal(parent(Λᵢ₋₁)); atol=wrap_eps(eltype(U)))), inds(Λᵢ₋₁)); dims=())
    end
    Γᵢ = if isnothing(Λᵢ₊₁)
        Vt
    else
        contract(Tensor(diag(pinv(Diagonal(parent(Λᵢ₊₁)); atol=wrap_eps(eltype(Vt)))), inds(Λᵢ₊₁)), Vt; dims=())
    end

    # Update the tensors in the tensor network
    replace!(ψ, tensors(ψ; at=sitel) => Γᵢ₋₁)
    replace!(ψ, tensors(ψ; at=siter) => Γᵢ)

    if canonize
        canonize!(ψ; normalize)
    else
        normalize && normalize!(ψ, collect((sitel, siter)))
    end

    return ψ
end

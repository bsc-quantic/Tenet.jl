# This file defines the "Ansatz" interface; i.e. Tensor Networks with a fixed structure.

struct AnsatzInterface end

function hasinterface(::AnsatzInterface, T::Type)
    hasmethod(lanes, Tuple{T}) || return false
    hasmethod(bonds, Tuple{T}) || return false
    hasmethod(tensors, Tuple{@NamedTuple{at::L} where {L<:AbstractLane},T}) || return false
    hasmethod(inds, Tuple{NamedTuple{bond::B} where {B<:Bond},T}) || return false
    return true
end

abstract type AnsatzTrait end
struct IsAnsatz <: AnsatzTrait end
struct WrapsAnsatz <: AnsatzTrait end
struct NotAnsatz <: AnsatzTrait end

function trait(::AnsatzInterface, ::T) where {T}
    if hasmethod(unwrap, Tuple{AnsatzInterface,T})
        return WrapsAnsatz()
    elseif hasinterface(AnsatzInterface(), T)
        return IsAnsatz()
    else
        return NotAnsatz()
    end
end

# required methods
"""
    lanes(tn)

Return the lanes of a Tensor Network.
"""
function lanes end

lanes(tn) = lanes(tn, trait(AnsatzInterface(), tn))
lanes(tn, ::WrapsAnsatz) = lanes(unwrap(AnsatzInterface(), tn))

"""
    bonds(tn)

Return the bonds of a Tensor Network.
"""
function bonds end

bonds(tn) = bonds(tn, trait(AnsatzInterface(), tn))
bonds(tn, ::WrapsAnsatz) = bonds(unwrap(AnsatzInterface(), tn))

"""
    tensors(tn; at::Lane)

Return the [`Tensor`](@ref) linked to a [`Lane`](@ref).
"""
tensors(kwargs::@NamedTuple{at::L}, tn) where {L<:Lane} = tensors(kwargs, tn, trait(AnsatzInterface(), tn))
tensors(kwargs::@NamedTuple{at::L}, tn, ::WrapsAnsatz) where {L<:Lane} = tensors(kwargs, unwrap(AnsatzInterface(), tn))

"""
    inds(tn; bond::Bond)

Return the index linked to a [`Bond`](@ref).
"""
inds(kwargs::@NamedTuple{bond::B}, tn) where {B<:Bond} = inds(kwargs, tn, trait(AnsatzInterface(), tn))
inds(kwargs::@NamedTuple{bond::B}, tn, ::WrapsAnsatz) where {B<:Bond} = inds(kwargs, unwrap(AnsatzInterface(), tn))

# alias
function inds(kwargs::@NamedTuple{bond::B}, tn) where {L<:AbstractLane,B<:Tuple{L,L}}
    inds((; bond=Bond(kwargs.bond...)), tn)
end

# TODO
# lanes(::@NamedTuple{at::Tensor}, tn::AbstractTensorNetwork)

# mutating methods
"""
    addlane!(tn, lane => tensor)

Link `lane` to `tensor`.
"""
function addlane! end

addlane!(tn, @nospecialize(p::Pair{<:Lane,<:Tensor})) = addlane!(tn, p.first, p.second)
addlane!(tn, lane::Lane, tensor::Tensor) = addlane!(tn, lane, tensor, trait(AnsatzInterface(), tn))
addlane!(tn, lane::Lane, tensor::Tensor, ::WrapsAnsatz) = addlane!(unwrap(AnsatzInterface(), tn), lane, tensor)

"""
    rmlane!(tn, lane)

Unlink `lane`.
"""
function rmlane! end

rmlane!(tn, lane::Lane) = rmlane!(tn, lane, trait(AnsatzInterface(), tn))
rmlane!(tn, lane::Lane, ::WrapsAnsatz) = rmlane!(unwrap(AnsatzInterface(), tn), lane)

"""
    addbond!(tn, bond => tensor)

Link `bond` to `tensor`.
"""
function addbond! end

addbond!(tn, @nospecialize(p::Pair{<:Bond,Symbol})) = addbond!(tn, bond, symbol)
addbond!(tn, bond::Bond, symbol::Symbol) = addbond!(tn, bond, symbol, trait(AnsatzInterface(), tn))
addbond!(tn, bond::Bond, symbol::Symbol, ::WrapsAnsatz) = addbond!(unwrap(AnsatzInterface(), tn), bond, symbol)

"""
    rmbond!(tn, bond)

Unlink `bond`.
"""
function rmbond! end

rmbond!(tn, bond::Bond) = rmbond!(tn, bond, trait(AnsatzInterface(), tn))
rmbond!(tn, bond::Bond, ::WrapsAnsatz) = rmbond!(unwrap(AnsatzInterface(), tn), bond)

# derived methods
Base.in(lane::Lane, tn::AbstractTensorNetwork) = haslane(tn, lane)
Base.in(bond::Bond, tn::AbstractTensorNetwork) = hasbond(tn, bond)

# optional methods
"""
    nlanes(tn)

Return the number of lanes of a Tensor Network.
"""
nlanes(tn) = nlanes(tn, trait(AnsatzInterface(), tn))

function nlanes(tn, ::IsAnsatz)
    @debug "Fallback to default implementation of `nlanes`"
    length(lanes(tn))
end

nlanes(tn, ::WrapsAnsatz) = nlanes(unwrap(AnsatzInterface(), tn))

"""
    haslane(tn, lane)

Return `true` if `lane` is in the Tensor Network.
"""
haslane(tn, lane) = haslane(tn, lane, trait(AnsatzInterface(), tn))

function haslane(tn, lane, ::IsAnsatz)
    @debug "Fallback to default implementation of `haslane`"
    lane ∈ lanes(tn)
end

haslane(tn, lane, ::WrapsAnsatz) = haslane(unwrap(AnsatzInterface(), tn), lane)

"""
    nbonds(tn)

Return the number of bonds of a Tensor Network.
"""
nbonds(tn) = nbonds(tn, trait(AnsatzInterface(), tn))

function nbonds(tn, ::IsAnsatz)
    @debug "Fallback to default implementation of `nbonds`"
    length(bonds(tn))
end

nbonds(tn, ::WrapsAnsatz) = nbonds(unwrap(AnsatzInterface(), tn))

"""
    hasbond(tn, bond)

Return `true` if `bond` is in the Tensor Network.
"""
hasbond(tn, bond) = hasbond(tn, bond, trait(AnsatzInterface(), tn))

function hasbond(tn, bond, ::IsAnsatz)
    @debug "Fallback to default implementation of `hasbond`"
    bond ∈ bonds(tn)
end

hasbond(tn, bond, ::WrapsAnsatz) = hasbond(unwrap(AnsatzInterface(), tn), bond)

"""
    neighbors(tn, lane)

Return the neighbors of a lane.
"""
Graphs.neighbors(tn::AbstractTensorNetwork, lane::Lane) = Graphs.neighbors(tn, lane, trait(AnsatzInterface(), tn))

function Graphs.neighbors(tn::AbstractTensorNetwork, lane::Lane, ::IsAnsatz)
    @debug "Fallback to default implementation of `neighbors` for `Ansatz`"
    [first(Iterators.filter(!=(lane), bond)) for bond in bonds(tn) if lane in bond]
end

function Graphs.neighbors(tn::AbstractTensorNetwork, lane::Lane, ::WrapsAnsatz)
    Graphs.neighbors(unwrap(AnsatzInterface(), tn), lane)
end

"""
    neighbors(tn, bond)

Return the neighbors of a bond.
"""
function Graphs.neighbors(tn::AbstractTensorNetwork, bond::Bond)
    a, b = Graphs.src(bond), Graphs.dst(bond)
    return filter!(x -> x != a && x != b, neighbors(tn, a) ∩ neighbors(tn, b))
end

"""
    isisometry(ψ, lane, bond; atol=1e-12)

Return `true` if the tensor at `lane` is an isometry pointing to `bond`.
"""
function isisometry(tn, lane, bond; atol::Real=1e-12)
    @assert haslane(bond, lane)
    @assert haslane(tn, lane)
    @assert hasbond(tn, bond)

    tensor = tensors(tn; at=lane)
    bondind = inds(tn; bond=bond)

    if isnothing(bondind)
        return isapprox(parent(contract(tensor, conj(tensor))), fill(true); atol)
    end

    inda, indb = gensym(:a), gensym(:b)
    a = replace(tensor, bondind => inda)
    b = replace(conj(tensor), bondind => indb)

    n = size(tensor, bondind)
    contracted = contract(a, b; out=[inda, indb])

    return isapprox(contracted, I(n); atol)
end

# TODO make an effect for this?
# NOTE in method == :svd the spectral weights are stored in a vector connected to the now virtual hyperindex!
function canonize_site!(tn, lane, bond; method=:qr, absorb=:dst)
    @assert haslane(bond, lane)
    @assert haslane(tn, lane)
    @assert hasbond(tn, bond)
    @assert absorb ∈ (nothing, :src, :source, :left, :dst, :destination, :right, :equal, :equally, :both)

    # A it the tensor where we perform the factorization, but B is also affected by the gauge transformation
    A = tensors(tn; at=lane)
    B = tensors(tn; at=only(filter(!=(lane), lanes(bond))))

    dirind = inds(tn; bond=bond)
    right_inds = Symbol[dirind]
    left_inds = filter(!=(dirind), vinds(A))

    tmpind = gensym(:tmp)
    if method === :svd
        # TODO use methods in Operations module when merged
        U, s, V = svd(A; left_inds, right_inds, virtualind=tmpind)

        # absorb singular values if specified
        if absorb ∈ (:src, :source, :left)
            U = contract(U, s; dims=[])
        elseif absorb ∈ (:dst, :destination, :right)
            V = contract(s, V; dims=[])
        elseif absorb ∈ (:equal, :equally, :both)
            U = contract(U, sqrt.(s); dims=[])
            V = contract(sqrt.(s), V; dims=[])
        end

        # contract V against next lane tensor
        V = contract(B, V)

        # rename back bond index
        U = replace(U, tmpind => dirind)
        V = replace(V, tmpind => dirind)
        s = replace(s, tmpind => dirind)

        # replace old tensors with new gauged ones
        replace!(tn, A => U)
        replace!(tn, B => V)

        # if singular values are not absorbed, connect it to the bond index forming an hyperindex
        isnothing(absorb) && push_inner!(tn, s)
    elseif method === :qr
        # QR always absorbs eigenvalues to the right
        @assert absorb ∈ (:dst, :destination, :right)

        # TODO use methods in Operations module when merged
        Q, R = qr(A; left_inds, right_inds, virtualind=tmpind)

        # contract R against next lane tensor
        R = contract(R, B)

        # rename back bond index
        Q = replace(Q, tmpind => dirind)
        R = replace(R, tmpind => dirind)

        # replace old tensors with new gauged ones
        replace!(tn, A => Q)
        replace!(tn, B => R)
    else
        throw(ArgumentError("Unknown factorization method=:$method"))
    end

    return tn
end

canonize_site(tn, lane, bond; kwargs...) = canonize_site!(copy(tn), lane, bond; kwargs...)

struct MissingSchmidtCoefficientsException <: Base.Exception
    bond::Bond
end

MissingSchmidtCoefficientsException(bond::Vector{<:AbstractLane}) = MissingSchmidtCoefficientsException(Bond(bond...))

function Base.showerror(io::IO, e::MissingSchmidtCoefficientsException)
    return print(io, "Can't access the spectrum on $(e.bond)")
end

"""
    tensors(tn; bond)

Return the [`Tensor`](@ref) in a virtual bond between two [`AbstractLane`](@ref)s in a Tensor Network.

# Notes

  - If the Tensor Network is in the canonical form, Tenet stores the Schmidt coefficients of the bond in a vector connected to the bond hyperedge between the two sites and the vector.
  - If the bond contains no Schmidt coefficients, this method will throw a `MissingSchmidtCoefficientsException`.
"""
function tensors(kwargs::NamedTuple{(:bond,)}, tn)
    vind = inds(tn; bond=kwargs.bond)
    tensor = tensors(tn; withinds=[vind])
    isempty(tensor) && throw(MissingSchmidtCoefficientsException(kwargs.bond))
    return only(tensor)
end

# TODO make an effect for this
"""
    absorb!(tn; bond=Bond(lane1, lane2), targetlane)

For a given Tensor Network, contract the singular values Λ located in the bond between lanes `lane1` and `lane2`.

# Keyword arguments

    - `bond` The bond between the singular values tensor and the tensors to be contracted.
    - `dir` The direction of the contraction. Defaults to `:left`.
"""
function absorb!(tn, bond, targetlane)
    @assert haslane(bond, targetlane)
    @assert haslane(tn, targetlane)
    @assert hasbond(tn, bond)

    # retrieve Λ tensor
    Λ = tensors(tn; bond)
    isnothing(Λ) && return tn

    # absorb to the target tensor
    Γ = tensors(tn; at=targetlane)
    replace!(tn, Γ => contract(Γ, Λ; dims=()))

    # remove Λ from the tensor network
    delete_inner!(tn, Λ)

    return tn
end

"""
    absorb(tn, args...; kwargs...)

Non-mutating version of [`absorb!`](@ref).
"""
absorb(tn, args...; kwargs...) = absorb!(copy(tn), args...; kwargs...)

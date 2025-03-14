# This file defines the "Ansatz" interface; i.e. Tensor Networks with a fixed structure.

struct AnsatzInterface end

function hasinterface(::AnsatzInterface, T::Type)
    hasmethod(lanes, Tuple{T}) || return false
    hasmethod(bonds, Tuple{T}) || return false
    hasmethod(tensors, Tuple{NamedTuple{(:at,)},T}) || return false
    hasmethod(inds, Tuple{NamedTuple{(:bond,)},T}) || return false
    return true
end

abstract type AnsatzTrait end
struct IsAnsatz <: AnsatzTrait end
struct WrapsAnsatz <: AnsatzTrait end
struct NotAnsatz <: AnsatzTrait end

function trait(::AnsatzInterface, ::T) where {T}
    if hasinterface(AnsatzInterface(), T)
        return IsAnsatz()
    elseif hasmethod(unwrap, Tuple{AnsatzInterface,T})
        return WrapsAnsatz()
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
tensors(kwargs::@NamedTuple{at::L}, tn) where {L<:Lane} = tensors(kwargs, trait(AnsatzInterface(), tn))
tensors(kwargs::@NamedTuple{at::L}, tn, ::WrapsAnsatz) where {L<:Lane} = tensors(kwargs, unwrap(AnsatzInterface(), tn))

"""
    inds(tn; bond::Bond)

Return the index linked to a [`Bond`](@ref).
"""
inds(kwargs::@NamedTuple{bond::B}, tn) where {B<:Bond} = inds(kwargs, trait(AnsatzInterface(), tn))
inds(kwargs::@NamedTuple{bond::B}, tn, ::WrapsAnsatz) where {B<:Bond} = inds(kwargs, unwrap(AnsatzInterface(), tn))

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

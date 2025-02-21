# This file defines the "Ansatz" interface; i.e. Tensor Networks with a fixed structure.

struct AnsatzInterface end

function hasinterface(::AnsatzInterface, T::Type)
    hasmethod(lanes, Tuple{T}) || return false
    hasmethod(bonds, Tuple{T}) || return false
    hasmethod(tensors, Tuple{NamedTuple{(:at,)},T}) || return false
    hasmethod(inds, Tuple{NamedTuple{(:bond,)},T}) || return false
    return true
end

# required methods
"""
    lanes(tn)

Return the lanes of a Tensor Network.
"""
function lanes end

lanes(tn::AbstractTensorNetwork) = lanes(tn, Wraps(AnsatzMixin, tn))
lanes(tn::AbstractTensorNetwork, ::Yes) = lanes(AnsatzMixin(tn))
lanes(tn::AbstractTensorNetwork, ::No) = throw(MethodError(lanes, (tn,)))

"""
    bonds(tn)

Return the bonds of a Tensor Network.
"""
function bonds end

bonds(tn::AbstractTensorNetwork) = bonds(tn, Wraps(AnsatzMixin, tn))
bonds(tn::AbstractTensorNetwork, ::Yes) = bonds(AnsatzMixin(tn))
bonds(tn::AbstractTensorNetwork, ::No) = throw(MethodError(bonds, (tn,)))

"""
    tensors(tn; at::Lane)

Return the [`Tensor`](@ref) linked to a [`Lane`](@ref).
"""
tensors(::@NamedTuple{at::L}, ::AbstractTensorNetwork) where {L<:Lane}

tensors(kwargs::@NamedTuple{at::L}, tn::AbstractTensorNetwork) where {L<:Lane} = tensors(kwargs, Wraps(AnsatzMixin, tn))
tensors(kwargs::@NamedTuple{at::L}, tn::AbstractTensorNetwork, ::Yes) where {L<:Lane} = tensors(kwargs, AnsatzMixin(tn))
function tensors(kwargs::@NamedTuple{at::L}, tn::AbstractTensorNetwork, ::No) where {L<:Lane}
    throw(MethodError(tensors, (kwargs, tn)))
end

"""
    inds(tn; bond::Bond)

Return the index linked to a [`Bond`](@ref).
"""
inds(::@NamedTuple{bond::B}, tn::AbstractTensorNetwork) where {B<:Bond}

inds(kwargs::@NamedTuple{bond::B}, tn::AbstractTensorNetwork) where {B<:Bond} = inds(kwargs, Wraps(AnsatzMixin, tn))
inds(kwargs::@NamedTuple{bond::B}, tn::AbstractTensorNetwork, ::Yes) where {B<:Bond} = inds(kwargs, AnsatzMixin(tn))
function inds(kwargs::@NamedTuple{bond::B}, tn::AbstractTensorNetwork, ::No) where {B<:Bond}
    throw(MethodError(inds, (kwargs, tn)))
end

# TODO
# lanes(::@NamedTuple{at::Tensor}, tn::AbstractTensorNetwork)

# optional methods
"""
    nlanes(tn)

Return the number of lanes of a Tensor Network.
"""
nlanes(tn) = nlanes(tn, Wraps(AnsatzMixin, tn))
nlanes(tn, ::Yes) = nlanes(AnsatzMixin(tn))
nlanes(tn, ::No) = length(lanes(tn))

"""
    haslane(tn, lane)

Return `true` if `lane` is in the Tensor Network.
"""
haslane(tn, lane) = haslane(tn, lane, Wraps(AnsatzMixin, tn))
haslane(tn, lane, ::Yes) = haslane(AnsatzMixin(tn), lane)
haslane(tn, lane, ::No) = lane ∈ lanes(tn)

"""
    nbonds(tn)

Return the number of bonds of a Tensor Network.
"""
hasbond(tn, lane) = hasbond(tn, lane, Wraps(AnsatzMixin, tn))
hasbond(tn, lane, ::Yes) = hasbond(AnsatzMixin(tn), lane)
hasbond(tn, lane, ::No) = lane ∈ lanes(tn)

"""
    hasbond(tn, bond)

Return `true` if `bond` is in the Tensor Network.
"""
hasbond(tn, bond) = bond ∈ bonds(tn)
hasbond(tn, bond, ::Yes) = hasbond(AnsatzMixin(tn), bond)
hasbond(tn, bond, ::No) = bond ∈ bonds(tn)

"""
    neighbors(tn, lane)

Return the neighbors of a lane.
"""
Graphs.neighbors(tn::AbstractTensorNetwork, lane::Lane) = Graphs.neighbors(tn, lane, Wraps(AnsatzMixin, tn))
Graphs.neighbors(tn::AbstractTensorNetwork, lane::Lane, ::Yes) = Graphs.neighbors(AnsatzMixin(tn), lane)
function Graphs.neighbors(tn::AbstractTensorNetwork, lane::Lane, ::No)
    [first(Iterators.filter(!=(lane), bond)) for bond in bonds(tn) if lane in bond]
end

"""
    neighbors(tn, bond)

Return the neighbors of a bond.
"""
function Graphs.neighbors(tn::AbstractTensorNetwork, bond::Bond)
    a, b = Graphs.src(bond), Graphs.dst(bond)
    return filter!(x -> x != a && x != b, neighbors(tn, a) ∩ neighbors(tn, b))
end

# mutating methods
"""
    addlane!(tn, lane => tensor)

Link `lane` to `tensor`.
"""
function addlane! end

addlane!(tn::AbstractTensorNetwork, @nospecialize(p::Pair{<:Lane,<:Tensor})) = addlane!(tn, p.first, p.second)
addlane!(tn::AbstractTensorNetwork, lane::Lane, tensor::Tensor) = addlane!(tn, lane, tensor, Wraps(AnsatzMixin, tn))
addlane!(tn::AbstractTensorNetwork, lane::Lane, tensor::Tensor, ::Yes) = addlane!(AnsatzMixin(tn), lane, tensor)
addlane!(tn::AbstractTensorNetwork, lane::Lane, tensor::Tensor, ::No) = throw(MethodError(addlane!, (tn, lane, tensor)))

"""
    rmlane!(tn, lane)

Unlink `lane`.
"""
function rmlane! end

rmlane!(tn::AbstractTensorNetwork, lane::Lane) = rmlane!(tn, lane, Wraps(AnsatzMixin, tn))
rmlane!(tn::AbstractTensorNetwork, lane::Lane, ::Yes) = rmlane!(AnsatzMixin(tn), lane)
rmlane!(tn::AbstractTensorNetwork, lane::Lane, ::No) = throw(MethodError(rmlane!, (tn, lane)))

"""
    addbond!(tn, bond => tensor)

Link `bond` to `tensor`.
"""
function addbond! end

addbond!(tn::AbstractTensorNetwork, @nospecialize(p::Pair{<:Bond,Symbol})) = addbond!(tn, bond, symbol)
addbond!(tn::AbstractTensorNetwork, bond::Bond, symbol::Symbol) = addbond!(tn, bond, symbol, Wraps(AnsatzMixin, tn))
addbond!(tn::AbstractTensorNetwork, bond::Bond, symbol::Symbol, ::Yes) = addbond!(AnsatzMixin(tn), bond, symbol)
addbond!(tn::AbstractTensorNetwork, bond::Bond, symbol::Symbol, ::No) = throw(MethodError(addbond!, (tn, bond, symbol)))

"""
    rmbond!(tn, bond)

Unlink `bond`.
"""
function rmbond! end

rmbond!(tn::AbstractTensorNetwork, bond::Bond) = rmbond!(tn, bond, Wraps(AnsatzMixin, tn))
rmbond!(tn::AbstractTensorNetwork, bond::Bond, ::Yes) = rmbond!(AnsatzMixin(tn), bond)
rmbond!(tn::AbstractTensorNetwork, bond::Bond, ::No) = throw(MethodError(rmbond!, (tn, bond)))

# derived methods
Base.in(lane::Lane, tn::AbstractTensorNetwork) = haslane(tn, lane)
Base.in(bond::Bond, tn::AbstractTensorNetwork) = hasbond(tn, bond)

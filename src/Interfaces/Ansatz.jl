# This file defines the "Ansatz" interface; i.e. Tensor Networks with a fixed structure.

# required methods
"""
    lanes(tn)

Return the lanes of a Tensor Network.
"""
function lanes end

"""
    bonds(tn)

Return the bonds of a Tensor Network.
"""
function bonds end

"""
    tensors(tn; at::Lane)

Return the [`Tensor`](@ref) linked to a [`Lane`](@ref).
"""
tensors(::@NamedTuple{at::Lane}, tn::AbstractTensorNetwork)

"""
    inds(tn; bond::Bond)

Return the index linked to a [`Bond`](@ref).
"""
inds(::@NamedTuple{bond::Bond}, tn::AbstractTensorNetwork)

# TODO
# lanes(::@NamedTuple{at::Tensor}, tn::AbstractTensorNetwork)

# optional methods
"""
    nlanes(tn)

Return the number of lanes of a Tensor Network.
"""
nlanes(tn) = length(lanes(tn))

"""
    haslane(tn, lane)

Return `true` if `lane` is in the Tensor Network.
"""
haslane(tn, lane) = lane ∈ lanes(tn)

"""
    nbonds(tn)

Return the number of bonds of a Tensor Network.
"""
nbonds(tn) = length(bonds(tn))

"""
    hasbond(tn, bond)

Return `true` if `bond` is in the Tensor Network.
"""
hasbond(tn, bond) = bond ∈ bonds(tn)

"""
    neighbors(tn, lane)

Return the neighbors of a lane.
"""
function Graphs.neighbors(tn::AbstractTensorNetwork, lane::Lane)
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

"""
    rmlane!(tn, lane)

Unlink `lane`.
"""
function rmlane! end

"""
    addbond!(tn, bond => tensor)

Link `bond` to `tensor`.
"""
function addbond! end

"""
    rmbond!(tn, bond)

Unlink `bond`.
"""
function rmbond! end

# derived methods
Base.in(lane::Lane, tn::AbstractTensorNetwork) = haslane(tn, lane)
Base.in(bond::Bond, tn::AbstractTensorNetwork) = hasbond(tn, bond)

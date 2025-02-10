# This file defines the "Ansatz" interface; i.e. Tensor Networks with a fixed structure.

"""
    lanes(tn)

Return the lanes of a Tensor Network.
"""
function lanes end

"""
    tensors(tn; at::Lane)

Return the [`Tensor`](@ref) linked to a [`Lane`](@ref).
"""
:(tensorat)

# TODO
# """
#     lanes()
# """
# :(laneat)

nlanes(tn) = length(lanes(tn))

haslane(tn, lane) = lane ∈ lanes(tn)
Base.in(lane::Lane, tn::AbstractTensorNetwork) = haslane(tn, lane)

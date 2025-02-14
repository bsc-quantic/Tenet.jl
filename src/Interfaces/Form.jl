"""
    Form

Abstract type representing the canonical form trait of a Tensor Network.
"""
abstract type Form end

Base.copy(x::Form) = x

"""
    NonCanonical

[`Form`](@ref) trait representing a Tensor Network in a non-canonical form.
"""
struct NonCanonical <: Form end

"""
    MixedCanonical

[`Form`](@ref) trait representing a Tensor Network in the mixed-canonical form.

  - The orthogonality center is a [`Lane`](@ref) or a vector of [`Lane`](@ref)s.
  - The tensors to the left and right of the orthogonality center are isommetries pointing towards the orthogonality center.
"""
struct MixedCanonical <: Form
    orthog_center::Union{Lane,Vector{<:Lane}}
end

Base.copy(x::MixedCanonical) = MixedCanonical(copy(x.orthog_center))

"""
    Canonical

[`Form`](@ref) trait representing a Tensor Network in canonical form or Vidal gauge; i.e. the singular values matrix
``\\Lambda_i`` between each tensor ``\\Gamma_{i-1}`` and ``\\Gamma_i``.
"""
struct Canonical <: Form end

"""
    form(tn)

Return the canonical form of the Tensor Network.
"""
function form end

"""
    canonize!(tn, form)

Transform an Tensor Network into a canonical [`Form`](@ref).

See also: [`NonCanonical`](@ref), [`MixedCanonical`](@ref), [`Canonical`](@ref).
"""
function canonize! end

"""
    canonize(tn)

Like [`canonize!`](@ref), but returns a new Tensor Network instead of modifying the original one.
"""
canonize(tn::AbstractTensorNetwork, args...; kwargs...) = canonize!(deepcopy(tn), args...; kwargs...)

# canonize_site(tn::AbstractTensorNetwork, args...; kwargs...) = canonize_site!(deepcopy(tn), args...; kwargs...)

"""
    checkform(tn)

Check whether a Tensor Network fulfills the properties of the canonical form is in.
"""
function checkform end

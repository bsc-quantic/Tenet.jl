using DelegatorTraits
using QuantumTags
using QuantumTags: Link

"""
    CanonicalForm

Abstract type representing the canonical form trait of a Tensor Network.
"""
abstract type CanonicalForm end

"""
    CanonicalForm(tn)

Return the canonical form of the Tensor Network.
"""
CanonicalForm(tn) = CanonicalForm(tn, DelegatorTrait(Tangle(), tn))
CanonicalForm(tn, ::DelegateToField) = CanonicalForm(delegator(Tangle(), tn))
CanonicalForm(_, ::DontDelegate) = NonCanonical()

const form = CanonicalForm

Base.copy(x::CanonicalForm) = x

checkform(tn; kwargs...) = checkform(tn, CanonicalForm(tn); kwargs...)
function checkform(tn, form; kwargs...)
    @warn "No check for $(typeof(tn)) with $(typeof(form)) form"
    return true
end

"""
    NonCanonical

[`CanonicalForm`](@ref) trait representing a Tensor Network in a non-canonical form.
"""
struct NonCanonical <: CanonicalForm end

# shortcut
checkform(tn, ::NonCanonical; kwargs...) = true

"""
    MixedCanonical

[`CanonicalForm`](@ref) trait representing a Tensor Network in the mixed-canonical form.

  - The orthogonality center is a [`Site`](@ref) or a vector of [`Site`](@ref)s.
  - The tensors to the left and right of the orthogonality center are isommetries pointing towards the orthogonality center.
"""
struct MixedCanonical{OrthogCenter} <: CanonicalForm
    orthog_center::OrthogCenter
end

Base.copy(x::MixedCanonical) = MixedCanonical(copy(orthog_center(x)))
Base.:(==)(a::MixedCanonical, b::MixedCanonical) = orthog_center(a) == orthog_center(b)

orthog_center(x::MixedCanonical) = x.orthog_center

min_orthog_center(x::MixedCanonical) = minimum(orthog_center(x))
min_orthog_center(x::MixedCanonical{<:Site}) = orthog_center(x)

max_orthog_center(x::MixedCanonical) = maximum(orthog_center(x))
max_orthog_center(x::MixedCanonical{<:Site}) = orthog_center(x)

"""
    BondCanonical

[`CanonicalForm`](@ref) trait representing a Tensor Network in the bond-canonical form.

  - The orthogonality center is a [`Bond`](@ref).
  - The tensors to the left and right of the orthogonality center are isommetries pointing towards the orthogonality center.
"""
struct BondCanonical <: CanonicalForm
    orthog_center::Bond
end

Base.copy(x::BondCanonical) = BondCanonical(copy(x.orthog_center))
Base.:(==)(a::BondCanonical, b::BondCanonical) = a.orthog_center == b.orthog_center

orthog_center(x::BondCanonical) = x.orthog_center

"""
    VidalGauge

[`CanonicalForm`](@ref) trait representing a Tensor Network in canonical form or Vidal gauge; i.e. the singular values matrix
``\\Lambda_i`` between each tensor ``\\Gamma_{i-1}`` and ``\\Gamma_i``.
"""
struct VidalGauge <: CanonicalForm end

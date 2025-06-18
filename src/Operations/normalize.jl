using LinearAlgebra: norm
import LinearAlgebra: normalize, normalize!

normalize(tn::AbstractTangle; kwargs...) = normalize!(copy(tn); kwargs...)

# `AbstractProduct`
function normalize!(tn::AbstractProduct, p::Real=2)
    for tensor in tensors(tn)
        normalize!(tensor, p)
    end
    return tn
end

# `MixedCanonicalMPS`
function normalize!(tn::MixedCanonicalMPS, p::Real=2)
    _min_orthog_center = min_orthog_center(form(tn))
    canonize!(tn, MixedCanonical(_min_orthog_center))
    normalize!(tensor_at(tn, _min_orthog_center), p)
    return tn
end

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

# `MPS` / `MPO`
generic_mps_normalize!(tn, p::Real=2) = generic_mps_normalize!(tn, form(tn), p)

# TODO compute by contracting contracting against its dual and updating all tensors?
function generic_mps_normalize!(tn, ::NonCanonical, p::Real=2)
    canonize!(tn, MixedCanonical(sites(tn)))
    return generic_mps_normalize!(tn, p)
end

function generic_mps_normalize!(tn, ::MixedCanonical, p::Real=2)
    _min_orthog_center = min_orthog_center(form(tn))
    canonize!(tn, MixedCanonical(_min_orthog_center))
    normalize!(tensor_at(tn, _min_orthog_center), p)
    return tn
end

function generic_mps_normalize!(tn::MPS, ::BondCanonical, p::Real=2)
    _bond = orthog_center(form(tn))
    normalize!(tensor_at(tn, LambdaSite(_bond)), p)
    return tn
end

normalize!(tn::MPS, p::Real=2) = generic_mps_normalize!(tn, p)
normalize!(tn::MPO, p::Real=2) = generic_mps_normalize!(tn, p)

import LinearAlgebra: norm

# `AbstractProduct`
function norm(tn::AbstractProduct, p::Real=2)
    mapreduce(*, tensors(tn)) do tensor
        norm(parent(tensor), p) # TODO is this implemented?
    end
end

# function LinearAlgebra.opnorm(tn::ProductOperator; p::Real=2)
#     return mapreduce(*, tensors(tn)) do tensor
#         opnorm(parent(tensor), p)
#     end
# end

# `MPS` / `MPO`
generic_mps_norm(tn, p::Real=2) = generic_mps_norm(tn, form(tn), p)

# TODO compute by contracting against its dual?
function generic_mps_norm(tn, ::NonCanonical, p::Real=2)
    canonize!(tn, MixedCanonical(sites(tn)))
    return generic_mps_norm(tn, p)
end

function generic_mps_norm(tn, ::MixedCanonical, p::Real=2)
    _min_orthog_center = min_orthog_center(form(tn))
    canonize!(tn, MixedCanonical(_min_orthog_center))
    return norm(tensor_at(tn, _min_orthog_center), p)
end

function generic_mps_norm(tn, ::BondCanonical, p::Real=2)
    _bond = orthog_center(form(tn))
    return norm(tensor_at(tn, LambdaSite(_bond)), p)
end

function generic_mps_norm(tn, ::VidalGauge, p::Real=2; check=true)
    _norms = map(Iterators.filter(s -> s isa LambdaSite, Tangles.all_sites_iter(tn))) do s
        norm(tensor_at(tn, s), p)
    end
    if check
        @assert all(x -> x ≈ _norms[1], _norms) "All norms must be approximately equal"
    end
    return _norms[1]
end

norm(tn::MPS, p::Real=2) = generic_mps_norm(tn, p)
norm(tn::MPO, p::Real=2) = generic_mps_norm(tn, p)

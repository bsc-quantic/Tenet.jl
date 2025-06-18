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

# `MixedCanonicalMPS`
# TODO what if `orthog_center` is not a single site?
function norm(tn::MixedCanonicalMPS, p::Real=2)
    _min_orthog_center = min_orthog_center(form(tn))
    canonize!(tn, MixedCanonical(_min_orthog_center))
    norm(tensor_at(tn, _min_orthog_center), p)
end

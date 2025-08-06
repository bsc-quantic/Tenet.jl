using DelegatorTraits: fallback

function overlap end

""" Computes overlap <ϕ|ψ> between two states, ** conjugates the second input state ** """
function overlap(ψ, ϕ; kwargs...)
    fallback(overlap)
    ϕ = resetinds!(conj(ϕ))
    align!(ψ, :outputs, ϕ, :outputs)
    tn = GenericTensorNetwork()
    append!(tn, all_tensors(ψ))
    append!(tn, all_tensors(ϕ))
    return contract(tn; kwargs...)
end

function overlap(a::ProductState, b::ProductState)
    issetequal(sites(a), sites(b)) || throw(ArgumentError("Both `ProductStates` must have the same sites"))
    return mapreduce(*, sites(a)) do site
        dot(tensor_at(a, site), conj(tensor_at(b, site)))
    end
end

function overlap(a::ProductState, b::AbstractMPS)
    issetequal(sites(a), sites(b)) || throw(ArgumentError("Both `ProductStates` must have the same sites"))
    align!(a, :outputs, b, :outputs)

    _tensor = binary_einsum(tensor_at(a, site"1"), tensor_at(b, site"1"))
    for i in 2:nsites(a)
        _tensor = binary_einsum(_tensor, binary_einsum(tensor_at(a, site"$i"), tensor_at(b, site"$i")))
    end

    return _tensor
end

overlap(a::AbstractMPS, b::ProductState) = overlap(b, a)

function overlap(a::MPS, b::MPS)
    @argcheck nsites(a) == nsites(b)

    b = resetinds!(conj(b))
    align!(a, :outputs, b, :outputs)

    left_env = binary_einsum(tensor_at(a, site"1"), tensor_at(b, site"1"))

    # left-to-right contraction sweep
    for i in 2:nsites(a)
        left_env = binary_einsum(binary_einsum(left_env, tensor_at(a, site"$i")), tensor_at(b, site"$i"))
    end

    return left_env
end

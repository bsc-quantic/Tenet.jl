module TenetTensorOperationsExt

using Tenet
using Tenet: TensorOperationsBackend
using TensorOperations: TensorOperations

# TODO we need to check edge cases not supported by TensorOperations and either error or dispatch to a different backend

function Tenet.contract(::TensorOperationsBackend, a::Tensor, b::Tensor; dims=(∩(inds(a), inds(b))), out=nothing)
    ia = collect(inds(a))
    ib = collect(inds(b))
    i = ∩(dims, ia, ib)
    ic::Vector{Symbol} = if isnothing(out)
        setdiff(ia ∪ ib, i isa Base.AbstractVecOrTuple ? i : (i,))::Vector{Symbol}
    else
        out
    end

    data = TensorOperations.tensorcontract(
        Tuple(ic), parent(a), Tuple(inds(a)), false, parent(b), Tuple(inds(b)), false
    )
    return Tensor(data, ic)
end

function Tenet.contract(::TensorOperationsBackend, a::Tensor; dims=nonunique(inds(a)), out=nothing)
    ia = inds(a)
    i = ∩(dims, ia)

    ic::Vector{Symbol} = if isnothing(out)
        setdiff(ia, i isa Base.AbstractVecOrTuple ? i : (i,))
    else
        out
    end

    # TODO might fail on partial trace
    data = TensorOperations.tensortrace(Tuple(ic), parent(a), Tuple(inds(a)), false)
    return Tensor(data, ic)
end

function Tenet.contract!(::TensorOperationsBackend, c::Tensor, a::Tensor, b::Tensor)
    pA, pB, pAB = TensorOperations.contract_indices(Tuple(inds(a)), Tuple(inds(b)), Tuple(inds(c)))
    tensorcontract!(parent(c), parent(a), pA, false, parent(b), pB, false, pAB)
    return c
end

function Tenet.contract!(::TensorOperationsBackend, y::Tensor, x::Tensor)
    p, q = TensorOperations.trace_indices(Tuple(inds(x)), Tuple(inds(y)))
    TensorOperations.tensortrace!(parent(y), parent(x), p, q, false)
    return y
end

end

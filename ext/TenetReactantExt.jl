module TenetReactantExt

using Tenet
using EinExprs
using Reactant
const MLIR = Reactant.MLIR
const stablehlo = MLIR.Dialects.stablehlo

function Tenet.contract(
    a::Tensor{Ta,Na,Aa}, b::Tensor{Tb,Nb,Ab}; kwargs...
) where {Ta,Na,Aa<:Reactant.ConcreteRArray,Tb,Nb,Ab<:Reactant.ConcreteRArray}
    c = @invoke Tenet.contract(a::Tensor, b::Tensor; kwargs...)
    return Tensor(Reactant.ConcreteRArray(parent(c)), inds(c))
end

function Tenet.contract(a::Tensor{T,N,A}; kwargs...) where {T,N,A<:Reactant.ConcreteRArray}
    c = @invoke Tenet.contract(a::Tensor; kwargs...)
    return Tensor(Reactant.ConcreteRArray(parent(c)), inds(c))
end

function Tenet.contract(
    a::Tensor{Ta,Na,Aa}, b::Tensor{Tb,Nb,Ab}; dims=(∩(inds(a), inds(b))), out=nothing
) where {Ta,Na,Aa<:Reactant.TracedRArray,Tb,Nb,Ab<:Reactant.TracedRArray}
    ia = collect(inds(a))
    ib = collect(inds(b))
    i = ∩(dims, ia, ib)

    ic::Vector{Symbol} = if isnothing(out)
        setdiff(ia ∪ ib, i isa Base.AbstractVecOrTuple ? i : (i,))::Vector{Symbol}
    else
        out
    end

    T = Base.promote_eltype(a, b)
    mlirty = Reactant.MLIR.IR.Type(T)

    op_a = parent(a).mlir_data
    op_b = parent(b).mlir_data
    rsize = Tuple(i ∈ ia ? size(a, i) : size(b, i) for i in ic)
    result_0 = Reactant.MLIR.IR.TensorType(rsize, mlirty)
    einsum_config = Reactant.MLIR.IR.Attribute("$(join(ia)),$(join(ib))->$(join(ic))")

    result = Reactant.MLIR.IR.result(stablehlo.einsum(op_a, op_b; result_0, einsum_config))

    data = Reactant.TracedRArray{T,rsize,length(ic)}((), result)
    _res = Tensor(data, ic)
    return _res
end

function Tenet.contract(a::Tensor{T,N,A}; dims=nonunique(inds(a)), out=nothing) where {T,N,A<:Reactant.TracedRArray}
    ia = inds(a)
    i = ∩(dims, ia)

    ic::Vector{Symbol} = if isnothing(out)
        setdiff(ia, i isa Base.AbstractVecOrTuple ? i : (i,))
    else
        out
    end

    mlirty = Reactant.MLIR.IR.Type(T)

    operand = parent(a).mlir_data
    rsize = Tuple(size(a, i) for i in ic)
    result_0 = Reactant.MLIR.IR.TensorType(rsize, mlirty)
    einsum_config = Reactant.MLIR.IR.Attribute("$(join(ia))->$(join(ic))")

    result = Reactant.MLIR.IR.result(stablehlo.unary_einsum(operand; result_0, einsum_config))

    data = Reactant.TracedRArray{T,rsize,length(ic)}((), result)
    return Tensor(data, ic)
end

end

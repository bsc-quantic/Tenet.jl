module TenetReactantExt

using Tenet
using EinExprs
using Reactant
const MLIR = Reactant.MLIR
const stablehlo = MLIR.Dialects.stablehlo

const Enzyme = Reactant.Enzyme

function Reactant.make_tracer(
    seen::IdDict, prev::RT, path::Tuple, mode::Reactant.TraceMode; kwargs...
) where {RT<:Tensor}
    tracedata = Reactant.make_tracer(seen, parent(prev), Reactant.append_path(path, 1), mode; kwargs...)
    return Tensor(tracedata, inds(prev))
end

function Reactant.make_tracer(seen::IdDict, prev::TensorNetwork, path::Tuple, mode::Reactant.TraceMode; kwargs...)
    tracetensors = map(enumerate(tensors(prev))) do (i, tensor)
        Reactant.make_tracer(seen, tensor, Reactant.append_path(path, i), mode; kwargs...)
    end
    return TensorNetwork(tracetensors)
end

Reactant.traced_getfield(x::TensorNetwork, i::Int) = tensors(x)[i]

function Reactant.make_tracer(seen::IdDict, prev::Quantum, path::Tuple, mode::Reactant.TraceMode; kwargs...)
    tracetn = Reactant.make_tracer(seen, TensorNetwork(prev), Reactant.append_path(path, :tn), mode; kwargs...)
    return Quantum(tracetn, copy(prev.sites))
end

function Reactant.make_tracer(seen::IdDict, prev::Tenet.Product, path::Tuple, mode::Reactant.TraceMode; kwargs...)
    tracequantum = Reactant.make_tracer(seen, Quantum(prev), Reactant.append_path(path, :super), mode; kwargs...)
    return Tenet.Product(tracequantum)
end

function Reactant.make_tracer(seen::IdDict, prev::Tenet.Chain, path::Tuple, mode::Reactant.TraceMode; kwargs...)
    tracequantum = Reactant.make_tracer(seen, Quantum(prev), Reactant.append_path(path, :super), mode; kwargs...)
    return Tenet.Chain(tracequantum, boundary(prev))
end

function Reactant.push_val!(ad_inputs, x::TensorNetwork, path)
    @assert length(path) == 2
    @assert path[2] === 1

    x = parent(tensors(x)[path[1]]).mlir_data

    return push!(ad_inputs, x)
end

function Reactant.set!(x::TensorNetwork, path, tostore; emptypath=false)
    @assert length(path) == 2
    @assert path[2] === 1

    x = parent(tensors(x)[path[1]])
    x.mlir_data = tostore

    if emptypath
        x.paths = ()
    end
end

function Reactant.set_act!(inp::Enzyme.Annotation{TensorNetwork}, path, reverse, tostore; emptypath=false)
    @assert length(path) == 2
    @assert path[2] === 1

    x = if inp isa Enzyme.Active
        inp.val
    else
        inp.dval
    end

    x = parent(tensors(x)[path[1]])
    x.mlir_data = tostore

    if emptypath
        x.paths = ()
    end
end

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

    counter = Tenet.IndexCounter()
    translator = Dict(i => only(string(Tenet.nextindex!(counter))) for i in ia ∪ ib)

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

    tia = Char[translator[i] for i in ia]
    tib = Char[translator[i] for i in ib]
    tic = Char[translator[i] for i in ic]
    einsum_config = Reactant.MLIR.IR.Attribute("$(join(tia)),$(join(tib))->$(join(tic))")

    result = Reactant.MLIR.IR.result(stablehlo.einsum(op_a, op_b; result_0, einsum_config))

    data = Reactant.TracedRArray{T,length(ic)}((), result, rsize)
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

    data = Reactant.TracedRArray{T,length(ic)}((), result, rsize)
    return Tensor(data, ic)
end

end

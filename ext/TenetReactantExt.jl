module TenetReactantExt

using Tenet
using EinExprs
using Reactant
using Reactant: TracedRArray
const MLIR = Reactant.MLIR
const stablehlo = MLIR.Dialects.stablehlo

const Enzyme = Reactant.Enzyme

function Reactant.make_tracer(
    seen, @nospecialize(prev::RT), path::Tuple, mode::Reactant.TraceMode; kwargs...
) where {RT<:Tensor}
    tracedata = Reactant.make_tracer(seen, parent(prev), Reactant.append_path(path, :data), mode; kwargs...)
    return Tensor(tracedata, copy(inds(prev)))
end

function Reactant.make_tracer(seen, prev::TensorNetwork, path::Tuple, mode::Reactant.TraceMode; kwargs...)
    tracetensors = Vector{Tensor}(undef, Tenet.ntensors(prev))
    for (i, tensor) in enumerate(tensors(prev))
        tracetensors[i] = Reactant.make_tracer(seen, tensor, Reactant.append_path(path, i), mode; kwargs...)
    end
    return TensorNetwork(tracetensors)
end

Reactant.traced_getfield(x::TensorNetwork, i::Int) = tensors(x)[i]

function Reactant.make_tracer(seen, prev::Quantum, path::Tuple, mode::Reactant.TraceMode; kwargs...)
    tracetn = Reactant.make_tracer(seen, TensorNetwork(prev), Reactant.append_path(path, :tn), mode; kwargs...)
    return Quantum(tracetn, copy(prev.sites))
end

function Reactant.make_tracer(seen, prev::Ansatz, path::Tuple, mode::Reactant.TraceMode; kwargs...)
    tracetn = Reactant.make_tracer(seen, Quantum(prev), Reactant.append_path(path, :tn), mode; kwargs...)
    return Ansatz(tracetn, copy(Tenet.lattice(prev)))
end

# TODO try rely on generic fallback for ansatzes
function Reactant.make_tracer(seen, prev::Tenet.Product, path::Tuple, mode::Reactant.TraceMode; kwargs...)
    tracetn = Reactant.make_tracer(seen, Ansatz(prev), Reactant.append_path(path, :tn), mode; kwargs...)
    return Tenet.Product(tracetn)
end

function Reactant.make_tracer(
    seen, prev::A, path::Tuple, mode::Reactant.TraceMode; kwargs...
) where {A<:Tenet.AbstractMPO}
    tracetn = Reactant.make_tracer(seen, Ansatz(prev), Reactant.append_path(path, :tn), mode; kwargs...)
    return A(tracetn, copy(form(prev)))
end

function Reactant.create_result(@nospecialize(tocopy::Tensor), @nospecialize(path), result_stores)
    data = Reactant.create_result(parent(tocopy), Reactant.append_path(path, :data), result_stores)
    return :($Tensor($data, $(copy(inds(tocopy)))))
end

function Reactant.create_result(tocopy::TensorNetwork, @nospecialize(path), result_stores)
    elems = map(1:Tenet.ntensors(tocopy)) do i
        Reactant.create_result(tensors(tocopy)[i], Reactant.append_path(path, i), result_stores)
    end
    return :($TensorNetwork([$(elems...)]))
end

function Reactant.create_result(tocopy::Quantum, @nospecialize(path), result_stores)
    tn = Reactant.create_result(TensorNetwork(tocopy), Reactant.append_path(path, :tn), result_stores)
    return :($Quantum($tn, $(copy(tocopy.sites))))
end

function Reactant.create_result(tocopy::Ansatz, @nospecialize(path), result_stores)
    tn = Reactant.create_result(Quantum(tocopy), Reactant.append_path(path, :tn), result_stores)
    return :($Ansatz($tn, $(copy(Tenet.lattice(tocopy)))))
end

# TODO try rely on generic fallback for ansatzes
function Reactant.create_result(tocopy::Tenet.Product, @nospecialize(path), result_stores)
    tn = Reactant.create_result(Ansatz(tocopy), Reactant.append_path(path, :tn), result_stores)
    return :($(Tenet.Product)($tn))
end

function Reactant.create_result(tocopy::A, @nospecialize(path), result_stores) where {A<:Tenet.AbstractMPO}
    tn = Reactant.create_result(Ansatz(tocopy), Reactant.append_path(path, :tn), result_stores)
    return :($A($tn, $(Tenet.form(tocopy))))
end

function Reactant.push_val!(ad_inputs, x::TensorNetwork, path)
    @assert length(path) == 2
    @assert path[2] === :data

    x = parent(tensors(x)[path[1]]).mlir_data

    return push!(ad_inputs, x)
end

function Reactant.set!(x::TensorNetwork, path, tostore; emptypath=false)
    @assert length(path) == 2
    @assert path[2] === :data

    x = parent(tensors(x)[path[1]])
    x.mlir_data = tostore

    if emptypath
        x.paths = ()
    end
end

function Reactant.set_act!(inp::Enzyme.Annotation{TensorNetwork}, path, reverse, tostore; emptypath=false)
    @assert length(path) == 2
    @assert path[2] === :data

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

# function Tenet.contract(
#     a::Tensor{Ta,Na,Aa}, b::Tensor{Tb,Nb,Ab}; kwargs...
# ) where {Ta,Na,Aa<:Reactant.ConcreteRArray,Tb,Nb,Ab<:Reactant.ConcreteRArray}
#     c = @invoke Tenet.contract(a::Tensor, b::Tensor; kwargs...)
#     return Tensor(Reactant.ConcreteRArray(parent(c)), inds(c))
# end

# function Tenet.contract(a::Tensor{T,N,A}; kwargs...) where {T,N,A<:Reactant.ConcreteRArray}
#     c = @invoke Tenet.contract(a::Tensor; kwargs...)
#     return Tensor(Reactant.ConcreteRArray(parent(c)), inds(c))
# end

# NOTE `@reactant_override` no longer exists
# @reactant_override function Base.:(-)(a::Tenet.Tensor, b::Tenet.Tensor)
#     issetequal(inds(a), inds(b)) || throw(ArgumentError("indices must be equal"))
#     perm = __find_index_permutation(inds(a), inds(b))
#     return Tensor(parent(a) - permutedims(parent(b), perm), inds(a))
# end

function Tenet.contract(
    a::Tensor{Ta,Na,TracedRArray{Ta,Na}}, b::Tensor{Tb,Nb,TracedRArray{Tb,Nb}}; dims=(∩(inds(a), inds(b))), out=nothing
) where {Ta,Na,Tb,Nb}
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

    data = TracedRArray{T,length(ic)}((), result, rsize)
    _res = Tensor(data, ic)
    return _res
end

function Tenet.contract(a::Tensor{T,N,TracedRArray{T,N}}; dims=nonunique(inds(a)), out=nothing) where {T,N}
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

    data = TracedRArray{T,length(ic)}((), result, rsize)
    return Tensor(data, ic)
end

Tenet.contract(a::Tensor, b::Tensor{T,N,TracedRArray{T,N}}; kwargs...) where {T,N} = contract(b, a; kwargs...)
function Tenet.contract(a::Tensor{Ta,Na,TracedRArray{Ta,Na}}, b::Tensor{Tb,Nb}; kwargs...) where {Ta,Na,Tb,Nb}
    # TODO change to `Ops.constant` when Ops PR lands in Reactant
    # apparently `promote_to` doesn't do the transpostion for converting from column-major (Julia) to row-major layout (MLIR)
    # currently, we call permutedims manually
    return contract(
        a,
        Tensor(Reactant.promote_to(TracedRArray{Tb,Nb}, permutedims(parent(b), collect(Nb:-1:1))), inds(b));
        kwargs...,
    )
end

end

module TenetReactantExt

using Tenet
using EinExprs
using Reactant
using Reactant: TracedRArray, TracedRNumber, @reactant_overlay
const MLIR = Reactant.MLIR
const stablehlo = MLIR.Dialects.stablehlo

const Enzyme = Reactant.Enzyme

# we specify `mode` and `track_numbers` types due to ambiguity
Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(_::Type{Tensor}), seen, mode::Reactant.TraceMode, @nospecialize(track_numbers::Type)
)
    return Tensor
end

Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(_::Type{Tensor{T}}), seen, mode::Reactant.TraceMode, @nospecialize(track_numbers::Type)
) where {T}
    return Tensor{TracedRNumber{T}}
end

Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(_::Type{Tensor{T,N}}), seen, mode::Reactant.TraceMode, @nospecialize(track_numbers::Type)
) where {T,N}
    return Tensor{TracedRNumber{T,N}}
end

Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(_::Type{Tensor{T,N,A}}), seen, mode::Reactant.TraceMode, @nospecialize(track_numbers::Type)
) where {T,N,A}
    A_traced = Reactant.traced_type_inner(A, seen, mode, track_numbers)
    T_traced = eltype(A_traced)
    return Tensor{T_traced,N,A_traced}
end

# we specify `mode` and `track_numbers` types due to ambiguity
Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(T::Type{<:Tenet.AbstractTensorNetwork}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type)
)
    return T
end

function Reactant.make_tracer(seen, @nospecialize(prev::RT), @nospecialize(path), mode; kwargs...) where {RT<:Tensor}
    tracedata = Reactant.make_tracer(seen, parent(prev), Reactant.append_path(path, :data), mode; kwargs...)
    return Tensor(tracedata, inds(prev))
end

function Reactant.make_tracer(seen, prev::TensorNetwork, @nospecialize(path), mode; kwargs...)
    tracetensors = Vector{Tensor}(undef, Tenet.ntensors(prev))
    for (i, tensor) in enumerate(tensors(prev))
        tracetensors[i] = Reactant.make_tracer(seen, tensor, Reactant.append_path(path, i), mode; kwargs...)
    end
    return TensorNetwork(tracetensors)
end

Reactant.traced_getfield(x::TensorNetwork, i::Int) = tensors(x)[i]

function Reactant.make_tracer(seen, prev::Quantum, @nospecialize(path), mode; kwargs...)
    tracetn = Reactant.make_tracer(seen, TensorNetwork(prev), Reactant.append_path(path, :tn), mode; kwargs...)
    return Quantum(tracetn, copy(prev.sites))
end

function Reactant.make_tracer(seen, prev::Ansatz, @nospecialize(path), mode; kwargs...)
    tracetn = Reactant.make_tracer(seen, Quantum(prev), Reactant.append_path(path, :tn), mode; kwargs...)
    return Ansatz(tracetn, copy(Tenet.lattice(prev)))
end

# TODO try rely on generic fallback for ansatzes
function Reactant.make_tracer(seen, prev::Tenet.Product, @nospecialize(path), mode; kwargs...)
    tracetn = Reactant.make_tracer(seen, Ansatz(prev), Reactant.append_path(path, :tn), mode; kwargs...)
    return Tenet.Product(tracetn)
end

function Reactant.make_tracer(
    seen, @nospecialize(prev::A), @nospecialize(path), mode; kwargs...
) where {A<:Tenet.AbstractMPO}
    tracetn = Reactant.make_tracer(seen, Ansatz(prev), Reactant.append_path(path, :tn), mode; kwargs...)
    return A(tracetn, copy(form(prev)))
end

function Reactant.create_result(@nospecialize(tocopy::Tensor), @nospecialize(path), result_stores)
    data = Reactant.create_result(parent(tocopy), Reactant.append_path(path, :data), result_stores)
    return :($Tensor($data, $(inds(tocopy))))
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

function Reactant.TracedUtils.push_val!(ad_inputs, x::TensorNetwork, path)
    @assert length(path) == 2
    @assert path[2] === :data

    x = parent(tensors(x)[path[1]]).mlir_data

    return push!(ad_inputs, x)
end

function Reactant.TracedUtils.set!(x::TensorNetwork, path, tostore; emptypath=false)
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

Base.@nospecializeinfer @noinline function Tenet.contract(
    @nospecialize(a::Tensor{TracedRNumber{Ta},Na,TracedRArray{Ta,Na}}),
    @nospecialize(b::Tensor{TracedRNumber{Tb},Nb,TracedRArray{Tb,Nb}});
    kwargs...,
) where {Ta,Na,Tb,Nb}
    #     return Base.inferencebarrier(__contract_binary)(a, b; kwargs...)
    # end

    # function __contract_binary(@nospecialize(a), @nospecialize(b); kwargs...)
    dims = get(kwargs, :dims) do
        ∩(inds(a), inds(b))
    end
    out = get(kwargs, :out, nothing)

    ia, ib = collect(inds(a)), collect(inds(b))
    @assert allunique(ia) "can't perform unary einsum operations on binary einsum"
    @assert allunique(ib) "can't perform unary einsum operations on binary einsum"
    @assert dims ⊆ ia ∩ ib "`dims` must be a subset of the intersection of the indices of the two tensors"
    @assert isnothing(out) || out ⊆ ia ∪ ib "`out` must be a subset of the union of the indices of the two tensors"
    @assert isnothing(out) || allunique(out) "indices in `out` for a binary einsum must be unique (no repetitions)"

    # override `dims` if `out` is provided
    dims = !isnothing(out) ? setdiff(dims, out) : dims

    contracting_inds = ∩(dims, ia, ib)
    contracting_dimensions = if isempty(contracting_inds)
        (Int[], Int[])
    else
        (map(i -> findfirst(==(i), ia), contracting_inds), map(i -> findfirst(==(i), ib), contracting_inds))
    end

    batching_inds = setdiff(ia ∩ ib, dims)
    batching_dimensions = if isempty(batching_inds)
        (Int[], Int[])
    else
        (map(i -> findfirst(==(i), ia), batching_inds), map(i -> findfirst(==(i), ib), batching_inds))
    end

    result_inds = setdiff(ia, contracting_inds, batching_inds) ∪ setdiff(ib, contracting_inds, batching_inds)
    ic = vcat(batching_inds, result_inds)

    # TODO replace for `Ops.convert`/`adapt` when it's available (there can be problems with nested array structures)
    T = Base.promote_eltype(a, b)
    da = eltype(a) != T ? TracedRArray{Reactant.unwrapped_eltype(T),ndims(a)}(parent(a)) : parent(a)
    db = eltype(b) != T ? TracedRArray{Reactant.unwrapped_eltype(T),ndims(b)}(parent(b)) : parent(b)

    data = Reactant.Ops.dot_general(da, db; contracting_dimensions, batching_dimensions)

    # if `out` is provided, emit `stablehlo.transpose` to correct dimension order
    if !isnothing(out)
        data = Reactant.Ops.transpose(data, map(i -> findfirst(==(i), ic), out))
        ic = out
    end

    return Tensor(data, ic)
end

function Tenet.contract(
    @nospecialize(a::Tensor{TracedRNumber{T},N,TracedRArray{T,N}}); dims=nonunique(inds(a)), out=nothing
) where {T,N}
    error("compilation of unary einsum operations are not yet supported")
end

function Tenet.contract(
    @nospecialize(a::Tensor), @nospecialize(b::Tensor{TracedRNumber{T},N,TracedRArray{T,N}}); kwargs...
) where {T,N}
    contract(b, a; kwargs...)
end
function Tenet.contract(
    @nospecialize(a::Tensor{TracedRNumber{T},N,TracedRArray{T,N}}), @nospecialize(b::Tensor); kwargs...
) where {T,N}
    return contract(a, Tensor(Reactant.Ops.constant(parent(b)), inds(b)); kwargs...)
end

# fixes issue with default `conj(x::AbstractArray) = x` method from Base (it might be overlayed in Reactant.jl)
Base.conj(@nospecialize(x::Tensor{<:TracedRNumber,N,<:TracedRArray})) where {N} = x
function Base.conj(@nospecialize(x::Tensor{TracedRNumber{T},N,<:TracedRArray})) where {T<:Complex,N}
    Tensor(conj(parent(x)), inds(x))
end

end

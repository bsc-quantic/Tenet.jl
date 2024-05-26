using EinExprs
using AbstractTrees
using Reactant
using Reactant.MLIR.Dialects: stablehlo
using Cassette

function compile(path::EinExpr; inplace::Bool=false)
    expr = if inplace
        codegen(Val{:inplace}(), path)
    else
        codegen(Val{:outplace}(), path)
    end

    return eval(expr)
end

function codegen(::Val{:inplace}, path::EinExpr)
    ssa = IdDict(leave => i for (i, leave) in enumerate(Iterators.flatten([Leaves(path), Branches(path)])))

    args = map(Iterators.flatten([Leaves(path), Branches(path)])) do node
        i = ssa[node]
        :($(Symbol(:ssa, i)))
    end

    ssa_eincodes = map(Branches(path)) do branch
        @assert EinExprs.nargs(branch) == 2 "Only binary contractions are supported"

        a, b = EinExprs.args(branch)
        i, j = ssa[a], ssa[b]

        ssa_a = Symbol(:ssa, i)
        ssa_b = Symbol(:ssa, j)

        k = ssa[branch]
        ssa_c = Symbol(:ssa, k)

        ssa[branch] = k

        return :(contract!($ssa_c, $ssa_a, $ssa_b))
    end

    fname = gensym(:contract_compiled!)

    quote
        function $fname($(args...))
            $(ssa_eincodes...)
            return $(Symbol(:ssa, ssa[path]))
        end

        function $fname($(args...))
            $(map(Iterators.flatten([Leaves(path), Branches(path)])) do node
                i = ssa[node]
                name = Symbol(:ssa, i)
                :($name = Tensor($name, $(head(node))))
            end...)

            return $fname($(args...))
        end
    end
end

function codegen(::Val{:outplace}, path::EinExpr)
    ssa = IdDict(leave => i for (i, leave) in enumerate(Iterators.flatten([Leaves(path), Branches(path)])))

    args = map(Leaves(path)) do node
        i = ssa[node]
        :($(Symbol(:ssa, i)))
    end

    tensorargs = map(Leaves(path)) do node
        i = ssa[node]
        :($(Symbol(:ssa, i))::Tensor)
    end

    ssa_eincodes = map(Branches(path)) do branch
        @assert EinExprs.nargs(branch) == 2 "Only binary contractions are supported"

        a, b = EinExprs.args(branch)
        i, j = ssa[a], ssa[b]

        ssa_a = Symbol(:ssa, i)
        ssa_b = Symbol(:ssa, j)

        k = ssa[branch]
        ssa_c = Symbol(:ssa, k)

        ssa[branch] = k

        return :($ssa_c = contract($ssa_a, $ssa_b))
    end

    fname = gensym(:contract_compiled)

    quote
        function $fname($(tensorargs...))
            $(ssa_eincodes...)
            return $(Symbol(:ssa, ssa[path]))
        end

        function $fname($(args...))
            $(map(Leaves(path)) do node
                i = ssa[node]
                name = Symbol(:ssa, i)
                :($name = Tensor($name, $(head(node))))
            end...)

            return $fname($(args...))
        end
    end
end

function codegen(::Val{:grad}, path::EinExpr)
    ssa = IdDict(leave => i for (i, leave) in enumerate(Iterators.flatten([Leaves(path), Branches(path)])))

    args = map(Leaves(path)) do node
        i = ssa[node]
        :($(Symbol(:ssa, i)))
    end

    tensorargs = map(Leaves(path)) do node
        i = ssa[node]
        :($(Symbol(:ssa, i))::Tensor)
    end

    ssa_primal_eincodes = map(Branches(path)) do branch
        @assert EinExprs.nargs(branch) == 2 "Only binary contractions are supported"

        a, b = EinExprs.args(branch)
        i, j = ssa[a], ssa[b]

        ssa_a = Symbol(:ssa, i)
        ssa_b = Symbol(:ssa, j)

        k = ssa[branch]
        ssa_c = Symbol(:ssa, k)

        ssa[branch] = k

        return :($ssa_c = contract($ssa_a, $ssa_b))
    end

    ssa_reverse_eincodes = map(Branches(path; inverse=true)) do branch
        @assert EinExprs.nargs(branch) == 2 "Only binary contractions are supported"

        a, b = EinExprs.args(branch)
        i, j = ssa[a], ssa[b]

        ssa_a = Symbol(:ssa, i)
        ssa_b = Symbol(:ssa, j)
        ∇ssa_a = Symbol(:∇ssa, i)
        ∇ssa_b = Symbol(:∇ssa, j)

        k = ssa[branch]
        ∇ssa_c = Symbol(:∇ssa, k)

        return quote
            $∇ssa_a = contract($∇ssa_c, conj($ssa_b); out=inds($ssa_a))
            $∇ssa_b = contract($∇ssa_c, conj($ssa_a); out=inds($ssa_b))
        end
    end

    fname = gensym(:∇contract)

    code_init_tangent = let k = ssa[path]
        ssa_c = Symbol(:ssa, k)
        ∇ssa_c = Symbol(:∇ssa, k)

        quote
            $∇ssa_c = similar($ssa_c)
            $∇ssa_c[] = one(eltype($∇ssa_c))
        end
    end

    quote
        function $fname($(tensorargs...))
            # primal computation
            $(ssa_primal_eincodes...)

            # TODO set tangent of the output before backpropagation
            $code_init_tangent

            # reverse pass
            $(ssa_reverse_eincodes...)

            return (;
                primal=$(Symbol(:ssa, ssa[path])),
                grad=($(map(((i, _),) -> Symbol(:∇ssa, i), enumerate(Leaves(path)))...),),
            )
        end

        function $fname($(args...))
            $(map(Leaves(path)) do node
                i = ssa[node]
                name = Symbol(:ssa, i)
                :($name = Tensor($name, $(head(node))))
            end...)

            return $fname($(args...))
        end
    end
end

function codegen(::Val{:TensorNetwork_from_arrays}, tn::TensorNetwork)
    args = map(1:ntensors(tn)) do i
        Symbol(:arg_, i)
    end

    fname = gensym(:TensorNetwork_from_arrays)

    quote
        function $fname($(args...))
            deserialized_tensors = Tensor[$(map(enumerate(tensors(tn))) do (i, tensor)
                :(Tensor($(Symbol(:arg_, i)), $(inds(tensor))))
            end...)]

            return TensorNetwork(deserialized_tensors)
        end
    end
end

function contract(
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

function contract(a::Tensor{T,N,A}; dims=nonunique(inds(a)), out=nothing) where {T,N,A<:Reactant.TracedRArray}
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

Cassette.overdub(ctx::Reactant.TraceCtx, f::typeof(contract), args...; kwargs...) = f(args...; kwargs...)

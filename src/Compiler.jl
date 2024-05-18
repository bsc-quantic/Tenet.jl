using EinExprs
using AbstractTrees

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
        N = ndims(node)
        :($(Symbol(:ssa, i))) #::Tensor{$T,$N})
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

        # WARN hardcoded return type
        return :(contract!($ssa_c, $ssa_a, $ssa_b))
    end

    :(function $(gensym(:contract_compiled))($(args...)) # FIX this is hardcoded
        $(ssa_eincodes...)
        return $(Symbol(:ssa, ssa[path]))
    end)
end

function codegen(::Val{:outplace}, path::EinExpr)
    ssa = IdDict(leave => i for (i, leave) in enumerate(Iterators.flatten([Leaves(path), Branches(path)])))

    args = map(Leaves(path)) do node
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

        return :($ssa_c = contract($ssa_a, $ssa_b))
    end

    :(function $(gensym(:contract_compiled))($(args...)) # FIX this is hardcoded
        $(ssa_eincodes...)
        return $(Symbol(:ssa, ssa[path]))
    end)
end

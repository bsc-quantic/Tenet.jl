using EinExprs
using OMEinsum
using AbstractTrees

function compile(path::EinExpr)
    ssa = IdDict(leave => i for (i, leave) in enumerate(Leaves(path)))

    ssa_leaves = map(Leaves(path)) do leave
        i = ssa[leave]
        :($(Symbol(:ssa, i)) = parent(getindex(tn, $(head(leave))...)))
    end

    ssa_eincodes = map(Branches(path)) do branch
        @assert EinExprs.nargs(branch) == 2 "Only binary contractions are supported"

        a, b = args(branch)
        i, j = ssa[a], ssa[b]

        ssa_a = Symbol(:ssa, i)
        ssa_b = Symbol(:ssa, j)

        k = length(ssa) + 1
        ssa_c = Symbol(:ssa, k)

        ssa[branch] = k

        ixs = (tuple(head(a)...), tuple(head(b)...))
        iy = tuple(head(branch)...)
        eincode = StaticEinCode{Symbol,ixs,iy}()

        :($ssa_c = $eincode($ssa_a, $ssa_b))
    end

    :(function $(gensym(:contract_compiled))(tn::TensorNetwork)
        # initialize SSA with leaves
        $(ssa_leaves...)

        # compute branches
        $(ssa_eincodes...)

        # return Tensor(Zygote.hook(c̄ -> @show(c̄), $(Symbol(:ssa, ssa[path]))), $(head(path)))
        return Tensor($(Symbol(:ssa, ssa[path])), $(head(path)))
    end)
end

using Tenet
using Yao: Yao
using EinExprs
using AbstractTrees
using Distributed
using Dagger
using TimespanLogging
using KaHyPar

m = 10
circuit = Yao.EasyBuild.rand_google53(m);
H = Quantum(circuit)
ψ = Product(fill([1, 0], Yao.nqubits(circuit)))
qtn = merge(Quantum(ψ), H, Quantum(ψ)')
tn = Tenet.TensorNetwork(qtn)

contract_smaller_dims = 20
target_size = 24

Tenet.transform!(tn, Tenet.ContractSimplification())
path = einexpr(
    tn;
    optimizer=HyPar(;
        parts=2,
        imbalance=0.41,
        edge_scaler=(ind_size) -> 10 * Int(round(log2(ind_size))),
        vertex_scaler=(prod_size) -> 100 * Int(round(exp2(prod_size))),
    ),
);

max_dims_path = @show maximum(ndims, Branches(path))
flops_path = @show mapreduce(flops, +, Branches(path))
@show log10(flops_path)

grouppath = deepcopy(path);
function recursiveforeach!(f, expr)
    f(expr)
    return foreach(arg -> recursiveforeach!(f, arg), args(expr))
end
sizedict = merge(Iterators.map(i -> i.size, Leaves(path))...);
recursiveforeach!(grouppath) do expr
    merge!(expr.size, sizedict)
    if all(<(contract_smaller_dims) ∘ ndims, expr.args)
        empty!(expr.args)
    end
end

max_dims_grouppath = maximum(ndims, Branches(grouppath))
flops_grouppath = mapreduce(flops, +, Branches(grouppath))
targetinds = findslices(SizeScorer(), grouppath; size=2^(target_size));

subexprs = map(Leaves(grouppath)) do expr
    only(EinExprs.select(path, tuple(head(expr)...)))
end

addprocs(3)
@everywhere using Dagger, Tenet

disttn = Tenet.TensorNetwork(
    map(subexprs) do subexpr
        Tensor(
            distribute( # data
                parent(Tenet.contract(tn; path=subexpr)),
                Blocks([i ∈ targetinds ? 1 : 2 for i in head(subexpr)]...),
            ),
            head(subexpr), # inds
        )
    end,
)
@show Tenet.contract(disttn; path=grouppath)

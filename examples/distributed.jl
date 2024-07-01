using Yao: Yao
using Tenet
using EinExprs
using KaHyPar
using Random
using Distributed
using ClusterManagers
using AbstractTrees

n = 64
depth = 6

circuit = Yao.chain(n)

for _ in 1:depth
    perm = randperm(n)

    for (i, j) in Iterators.partition(perm, 2)
        push!(circuit, Yao.put((i, j) => Yao.EasyBuild.FSimGate(2π * rand(), 2π * rand())))
        # push!(circuit, Yao.control(n, i, j => Yao.phase(2π * rand())))
    end
end

H = Quantum(circuit)
ψ = zeros(Product, n)

tn = TensorNetwork(merge(Quantum(ψ), H, Quantum(ψ)'))
transform!(tn, Tenet.ContractSimplification())

path = einexpr(
    tn;
    optimizer=HyPar(;
        parts=2,
        imbalance=0.41,
        edge_scaler=(ind_size) -> 10 * Int(round(log2(ind_size))),
        vertex_scaler=(prod_size) -> 100 * Int(round(exp2(prod_size))),
    ),
)

@show maximum(ndims, Branches(path))
@show maximum(length, Branches(path)) * sizeof(eltype(tn)) / 1024^3

@show log10(mapreduce(flops, +, Branches(path)))

cutinds = findslices(SizeScorer(), path; size=2^24)
cuttings = [[i => dim for dim in 1:size(tn, i)] for i in cutinds]

# mock sliced path - valid for all slices
proj_inds = first.(cuttings)
slice_path = view(path.path, proj_inds...)

expr = Tenet.codegen(Val(:outplace), slice_path)

manager = SlurmManager(2 * 112 - 1)
addprocs(manager; cpus_per_task=1, exeflags="--project=$(Base.active_project())")
# @everywhere using LinearAlgebra
# @everywhere LinearAlgebra.BLAS.set_num_threads(2)

@everywhere using Tenet, EinExprs, IterTools, LinearAlgebra, Reactant, AbstractTrees
@everywhere tn = $tn
@everywhere slice_path = $slice_path
@everywhere cuttings = $cuttings
@everywhere expr = $expr

partial_results = map(enumerate(workers())) do (i, worker)
    Distributed.@spawnat worker begin
        # interleaved chunking without instantiation
        it = takenth(Iterators.drop(Iterators.product(cuttings...), i - 1), nworkers())

        f = @eval $expr
        mock_slice = view(tn, first(it)...)
        tensors′ = [
            Tensor(Reactant.ConcreteRArray(copy(parent(mock_slice[head(leaf)...]))), inds(mock_slice[head(leaf)...])) for leaf in Leaves(slice_path)
        ]
        g = Reactant.compile(f, Tuple(tensors′))

        # local reduction of chunk
        accumulator = zero(eltype(tn))

        for proj_inds in it
            slice = view(tn, proj_inds...)
            tensors′ = [
                Tensor(
                    Reactant.ConcreteRArray(copy(parent(mock_slice[head(leaf)...]))),
                    inds(mock_slice[head(leaf)...]),
                ) for leaf in Leaves(slice_path)
            ]
            res = only(g(tensors′...))

            # avoid OOM due to garbage accumulation
            GC.gc()

            accumulator += res
        end

        return accumulator
    end
end

@show result = sum(Distributed.fetch.(partial_results))

rmprocs(workers())

# Google's Quantum Advantage experiment

```@setup circuit
using CairoMakie
CairoMakie.activate!(type = "svg")
using NetworkLayout
```

!!! info "Dependencies 📦"
    This example uses `QuacIO` and `EinExprs` in combination with `Tenet`.
    Both packages can be found in [Quantic's registry](https://github.com/bsc-quantic/Registry) and can be installed in Pkg mode.

    ```julia
    add QuacIO EinExprs
    ```

    It also requires the circuit in `sycamore_m53_d10.qasm` file that can be found in [here](./sycamore_53_10_0.qasm).
    This is a shorter version of the real circuit used for the experiment.

In 2019, Google rushed to claim _quantum advantage_[^supremacy] for the first time ever [arute2019quantum](@cite)[villalonga2020establishing](@cite).
The article was highly criticized and one year later, it was disproved [gray2021hyper](@cite) by developing a better heuristic search for contraction path which provided a $\times 10^4$ speedup.

[^supremacy]: The first used term was _quantum supremacy_ although the community transitioned to _quantum advantage_ due to political reasons. However, Google now uses the term _beyond classical_. It is then not uncommon to find different terms to refer to the same thing: the moment in which quantum computers surpass classical computers on solving some problem.

Since then, several teams and companies have come and go, proposing and disproving several experiments. But in this example, we focus on the original Google experiment.

In short, the experiment consisted on sampling Random Quantum Circuits (RQC).
The state of the systems after these circuits follow a distribution similar, but **not equal** to the uniform distribution.
Due to noise and decoherence, the fidelity of quantum chips decrease with the circuit depth.
The complexity of contracting tensor networks grows with the circuit depth, but due to the fidelity of the physical realization being small, a very rough approximation can be used.
In the case of Google, they used _tensor slicing_ for projecting some expensive to contract indices.
Since the contribution of each quantum path is guessed to be similar, each slice should contribute a similar part, and by taking the same percentage of slices as the fidelity of the quantum experiment, we obtain a result with a similar fidelity.
If you want to read more on the topic, check out [boixo2018characterizing](@cite),[markov2018quantum](@cite).

Thanks to `Tenet`'s much cared design, the experiment can be replicated conceptually in less than 20loc.

```@example circuit
using QuacIO
using Tenet

_sites = [5, 6, 14, 15, 16, 17, 24, 25, 26, 27, 28, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 61, 62, 63, 64, 65, 66, 67, 72, 73, 74, 75, 76, 83, 84, 85, 94];

# load circuit and convert to `TensorNetwork`
circuit = QuacIO.parse(joinpath(@__DIR__, "sycamore_53_10_0.qasm"), format = QuacIO.Qflex(), sites = _sites);
tn = TensorNetwork(circuit)
tn = view(tn, [i => 1 for i in inds(tn, set=:open)]...)
plot(tn) # hide
```

In order to aid the contraction path optimization, we shrink the search space by using local transformations.

```@example circuit
# simplify Tensor Network by preemptively contracting trivial cases
tn = transform(tn, Tenet.RankSimplification)
plot(tn, layout=Stress()) # hide
```

Contraction path optimization is the focus of the [`EinExprs`](https://github.com/bsc-quantic/EinExprs.jl) package. For this example, we will use the `Greedy` algorithm which doesn't yield the optimal path but it's fast and reproducible.

```@example circuit
using EinExprs
path = einexpr(tn, optimizer = Greedy)
plot(path, layout=Stress()) # hide
```

Then, the indices to be sliced have to be selected. `EinExprs` provides us with the `findslices` algorithm (based in the `SliceFinder` algorithm of [cotengra](@cite)) to suggest candidate indices for slicing.

```julia
cuttings = [[i => dim for dim in 1:size(tn,i)] for i in findslices(FlopsScorer(), path, slices=100)]
```

Finally, the contraction of slices is parallelized using distributed workers and each contribution is summed to `result`.

```julia
using Distributed
using Iterators: product

addprocs(10)

@everywhere using Tenet, EinExprs
@everywhere tn = $tn
@everywhere path = $path

result = @distributed (+) for proj_inds in product(cuttings...)
    slice = view(tn, proj_inds...)

    for indices in contractorder(path)
        contract!(slice, indices)
    end

    tensors(slice) |> only
end
```

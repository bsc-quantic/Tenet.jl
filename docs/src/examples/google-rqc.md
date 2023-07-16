# Google's Quantum Advantage experiment

```@setup circuit
using CairoMakie
CairoMakie.activate!(type = "svg")
```

!!! danger "🚧 Broken code 🚧"
    There is a lot of work in progress, and this code may not work yet.
    Specifically, `slices` is not implemented yet.
    Take this code as an example of what we want to achieve.

!!! info "Dependencies 📦"
    This example uses `QuacIO` and `EinExprs` in combination with `Tenet`.
    Both packages can be found in [Quantic's registry](https://github.com/bsc-quantic/Registry) and can be installed in Pkg mode.

    ```julia
    add QuacIO EinExprs
    ```

    It also requires the circuit in `sycamore_m53_d10.qasm` file that can be found in ...
    This is a shorter version of the real circuit used for the experiment.

In 2019, Google rushed to claim _quantum advantage_[^1] for the first time ever.
The article was highly criticized and one year later, J. Gray et al.[^2] disproved the claim by developing a better heuristic search for contraction path that improved the previous cost by $\times 10^4$.

[^1]: The first used term was _quantum supremacy_ although the community transitioned to _quantum advantage_ due to political reasons. However, Google uses the term _beyond classical_. It is then not uncommon to find different terms to refer to the same thing: the moment in which quantum computers surpass classical computers on solving some problem.

[^2]: J. Gray et al. "Hyper-Optimized Tensor Network Contraction" (2020)

Since then, several teams and companies have come and go, proposing and disproving several experiments. But in this example, we focus on the original Google experiment.

The experiment consisted on sampling Random Quantum Circuits (RQC). The state of the systems after these circuits follow a distribution similar, but **not equal** to the uniform distribution.

...

Thanks to `Tenet`'s much cared design, the experiment can be replicated conceptually in less than 20loc.

```@example circuit
using QuacIO
using Tenet

_sites = [5, 6, 14, 15, 16, 17, 24, 25, 26, 27, 28, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 61, 62, 63, 64, 65, 66, 67, 72, 73, 74, 75, 76, 83, 84, 85, 94];

# load circuit and convert to `TensorNetwork`
circuit = QuacIO.parse(joinpath(@__DIR__, "sycamore_53_10_0.qasm"), format = QuacIO.Qflex(), sites = _sites);
tn = TensorNetwork(circuit)
plot(tn) # hide
```

```@example circuit
# simplify Tensor Network by preemptively contracting trivial cases
tn = transform(tn, Tenet.RankSimplification)
plot(tn) # hide
```

```julia
addprocs(10)
@everywhere using Tenet, EinExprs

# parallel stochastic contraction path search
@everywhere tn = $tn
path = @distributed (x -> minimum(flops, x...)) for _ in 1:100
    einexpr(tn, optimizer = Greedy)
end
```

```@example circuit
using EinExprs # hide
using NetworkLayout # hide
path = einexpr(tn, optimizer = Greedy) # hide
plot(path, layout=Stress()) # hide
```

```julia
using Distributed
using Iterators: product

addprocs(10)
@everywhere using Tenet, EinExprs

# parallel stochastic contraction path search
@everywhere tn = $tn
path = @distributed (x -> minimum(flops, x...)) for _ in 1:100
    einexpr(tn, optimizer = Greedy)
end

# parallel sliced contraction
# NOTE `slices` not implemented yet
cuttings = [[i => dim for dim in 1:size(tn,i)] for i in slices(path, n=10)]

@everywhere path = $path
res = @distributed (+) for proj_inds in product(cuttings...)
    slice = @view path[proj_inds...]
    contract(slice)
end
```

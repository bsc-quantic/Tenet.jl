# Google's Quantum Advantage experiment

!!! danger "🚧 Broken code 🚧"
    There is a lot of work in progress, and this code may not work yet.
    Specifically, `Quac.parse` is not yet merged into `master` branch and `slices` is not implemented yet.
    Take this code as an example of what we want to achieve.

!!! info "Dependencies 📦"
    This example uses `Quac` and `EinExprs` in combination with `Tenet`.
    Both packages can be found in [Quantic's registry](https://github.com/bsc-quantic/Registry) and can be installed in Pkg mode.

    ```
    add Quac EinExprs
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

```@example circuit
using Quac # hide
circuit = Quac.parse("sycamore_m53_d10.qasm") # hide
```

Thanks to `Tenet`'s much cared design, the experiment can be replicated conceptually in less than 20loc.

```julia
using Quac
using Tenet
using Distributed
using Iterators: product

addprocs(10)
@everywhere using Tenet, EinExprs

# load circuit and convert to `TensorNetwork`
circuit = Quac.parse("sycamore_m53_d10.qasm")
tn = TensorNetwork(circuit)

# simplify Tensor Network by preemptively contracting trivial cases
tn = transform(tn, RankSimplification)

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

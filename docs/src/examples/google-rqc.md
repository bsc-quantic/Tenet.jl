# Google's Quantum Advantage experiment

!!! danger "🚧 Broken code 🚧"
    There is a lot of work in progress, and this code may not work yet.
    Specifically, `Quac.parse` is not yet merged into the `master` branch and `slices` is not implemented yet.
    Take this code as an example of what we want to achieve.

```julia
using Quac
using Tenet
using Distributed
using Iterators: product

addprocs(10)

# load circuit and convert to `TensorNetwork`
circuit = Quac.parse("sycamore_m53_d10.qasm")
tn = TensorNetwork(circuit)

# simplify Tensor Network by preemptively contracting trivial cases
tn = transform(tn, RankSimplification)

# parallel stochastic contraction path search
path = @distributed (x -> minimum(flops, x...)) for _ in 1:100
    einexpr(tn, optimizer = Greedy)
end

# parallel sliced contraction
# NOTE `slices` not implemented yet
cuttings = [[i => dim for dim in 1:size(tn,i)] for i in slices(path, n=10)]

res = @distributed (+) for proj_inds in product(cuttings...)
    slice = @view path[proj_inds...]
    contract(slice)
end
```

# Google's Quantum Advantage experiment

!!! danger "🚧 Broken code 🚧"
    There is a lot of work in progress, and this code may not work yet.
    Take this code as an example of what we want to achieve.

```julia
using Quac
using Tenet
using Distributed

tn = TensorNetwork(RQC(Sycamore, depth=12))
path = einexpr(tn)
sliced_inds = [[i => dim for dim in 1:size(tn,i)] for i in slices(path, n=10)]

addprocs(10)

res = @distributed (+) for proj_inds in product(sliced_inds...)
    tn_slice = selectdim(tn, proj_inds…)
    contract(tn_slice)
end
```

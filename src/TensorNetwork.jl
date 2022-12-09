import OptimizedEinsum
import OptimizedEinsum: optimize, Greedy
using NamedDims: dimnames

abstract type TensorNetwork end

Base.summary(io::IO, x::TensorNetwork) = print(io, "$(length(x))-tensors $(typeof(x))")
Base.show(io::IO, tn::TensorNetwork) =
    print(io, "$(typeof(tn))(#tensors=$(length(tn)), #inds=$(length(keys(tn.ind_size))))")

Base.length(x::TensorNetwork) = length(tensors(x))

function tensors end

arrays(tn::TensorNetwork) = parent.(tensors(tn))

function inds end

openinds(tn::TensorNetwork) = filter(ind -> count(∋(ind) ∘ dimnames, values(tensors(tn))) == 1, inds(tn))

function optimize(opt, tn::TensorNetwork; output = openinds(tn))
    inputs = collect.(dimnames.(tensors(tn)))
    output = collect(output)
    size = tn.ind_size
    optimize(opt, inputs, output, size)
end

optimize(tn::TensorNetwork; kwargs...) = optimize(Greedy, tn; kwargs...)

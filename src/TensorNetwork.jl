import Base: length, show, summary
import OptimizedEinsum: optimize, Greedy
using NamedDims

struct TensorNetwork
    tensors::Vector{NamedDimsArray}
    ind_size::Dict{Symbol,Int}
end

Base.length(x::TensorNetwork) = length(x.tensors)

Base.summary(io::IO, x::TensorNetwork) = println(io, "$(length(x))-tensors TensorNetwork")

inds(tn::TensorNetwork) = keys(tn.ind_size)

openinds(tn::TensorNetwork) = filter(ind -> count(∋(ind) ∘ dimnames, values(tn.tensors)) == 1, inds(tn))

hyperinds(tn::TensorNetwork) = filter(ind -> count(∋(ind) ∘ dimnames, values(tn.tensors)) > 2, inds(tn))

function optimize(opt, tn::TensorNetwork; output=openinds(tn))
    inputs = dimnames.(tn.tensors)
    size = tn.ind_size
    optimize(opt, tn.tensors, output, size)
end

optimize(tn::TensorNetwork; kwargs...) = optimize(Greedy, tn; kwargs...)

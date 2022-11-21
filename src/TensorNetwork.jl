import Base: length, show, summary
using OptimizedEinsum
using NamedDims

struct TensorNetwork
    tensors::Dict{Int,NamedDimsArray}
end

Base.length(x::TensorNetwork) = length(x.tensors)

Base.summary(io::IO, x::TensorNetwork) = println(io, "$(length(x))-tensors TensorNetwork")

inds(tn::TensorNetwork) = ∪(dimnames.(values(tn.tensors)))

openinds(tn::TensorNetwork) = filter(ind -> count(∋(ind) ∘ dimnames, values(tn.tensors)) == 1, inds(tn))

hyperinds(tn::TensorNetwork) = filter(ind -> count(∋(ind) ∘ dimnames, values(tn.tensors)) > 2, inds(tn))
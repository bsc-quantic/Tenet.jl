import Base: length, show, summary
using OptimizedEinsum
using NamedDims

struct TensorNetwork
    tensors::Dict{Int,NamedDimsArray}
end

Base.length(x::TensorNetwork) = length(x.tensors)

Base.show(io::IO, x::TensorNetwork) = println(io, "$(length(x))-tensors $(typeof(x))")

inds(tn::TensorNetwork) = ∪(dimnames.(values(tn.tensors)))

hyperinds(tn::TensorNetwork) = filter(ind -> count(∋(ind) ∘ dimnames, values(tn.tensors)) > 2, inds(tn))
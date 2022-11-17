import Base: length, show, summary
using OptimizedEinsum

struct TensorNetwork
    tensors::Dict{Int,NamedDimsArray}
end

Base.length(x::TensorNetwork) = length(x.tensors)

Base.show(io::IO, x::TensorNetwork) = println(io, "$(length(x))-tensors $(typeof(x))")

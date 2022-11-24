import Base: length, show, summary
import OptimizedEinsum: optimize, Greedy
using NamedDims

"""
    TensorNetwork

A generic Tensor Network structure.
"""
struct TensorNetwork
    tensors::Vector{NamedDimsArray}
    ind_size::Dict{Symbol,Int}
    ind_map::Dict{Symbol,Set{Int}}

    function TensorNetwork(tensors::Vector{NamedDimsArray})
        ind_size = Dict{Symbol,Int}()
        ind_map = Dict{Symbol,Set{Int}}()
        for (i, tensor) in enumerate(tensors)
            for ind in dimnames(tensor)
                if ind in ind_map
                    if ind_size[ind] != size(tensor, ind)
                        throw(ArgumentError("size of index $ind in tensor #$i ($(size(tensor,ind))) does not match previous assigment ($(ind_size[ind]))"))
                    else
                        ind_size[ind]
                    end
                    ind_map[ind] ∪= [i]
                else
                    ind_map[ind] = Set([i])
                    ind_size[ind] = size(tensor, ind)
                end
            end
        end
        new(tensors, ind_size, ind_map)
    end
end

tensors(tn::TensorNetwork) = tn.tensors

arrays(tn::TensorNetwork) = parent.(tensors(tn))

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

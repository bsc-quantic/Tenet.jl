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

    function TensorNetwork(tensors)
        ind_size = Dict{Symbol,Int}()
        ind_map = Dict{Symbol,Set{Int}}()
        tn = new(Vector{NamedDimsArray}(), ind_size, ind_map)

        for tensor in tensors
            push!(tn, tensor)
        end

        return tn
    end
end

tensors(tn::TensorNetwork) = tn.tensors
tensors(tn::TensorNetwork, i) = tn.tensors[i]

arrays(tn::TensorNetwork) = parent.(tensors(tn))

Base.length(x::TensorNetwork) = length(x.tensors)

Base.summary(io::IO, x::TensorNetwork) = println(io, "$(length(x))-tensors TensorNetwork")
Base.show(io::IO, tn::TensorNetwork) = println(io, "TensorNetwork(#tensors=$(length(tn)), #inds=$(length(keys(tn.ind_size))))")

inds(tn::TensorNetwork) = keys(tn.ind_size)

openinds(tn::TensorNetwork) = filter(ind -> count(∋(ind) ∘ dimnames, values(tn.tensors)) == 1, inds(tn))

hyperinds(tn::TensorNetwork) = filter(ind -> count(∋(ind) ∘ dimnames, values(tn.tensors)) > 2, inds(tn))

function Base.push!(tn::TensorNetwork, tensor::NamedDimsArray)
    i = maximum(Iterators.flatten(collect.(values(tn.ind_size)))) + 1

    for ind in dimnames(tensor)
        if ind in keys(tn.ind_map)
            if tn.ind_size[ind] != size(tensor, ind)
                throw(ArgumentError("size of index $ind in tensor #$i ($(size(tensor,ind))) does not match previous assigment ($(tn.ind_size[ind]))"))
            else
                tn.ind_size[ind]
            end
            tn.ind_map[ind] = tn.ind_map[ind] ∪ [i]
        else
            tn.ind_map[ind] = Set([i])
            tn.ind_size[ind] = size(tensor, ind)
        end
    end

    push!(tn.tensors, tensor)
end

function optimize(opt, tn::TensorNetwork; output=openinds(tn))
    inputs = collect.(dimnames.(tn.tensors))
    output = collect(output)
    size = tn.ind_size
    optimize(opt, inputs, output, size)
end

optimize(tn::TensorNetwork; kwargs...) = optimize(Greedy, tn; kwargs...)

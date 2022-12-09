import Random: rand
using NamedDims

"""
    GenericTensorNetwork

A Tensor Network with arbitrary structure.
"""
struct GenericTensorNetwork <: TensorNetwork
    tensors::Vector{NamedDimsArray}
    ind_size::Dict{Symbol,Int}
    ind_map::Dict{Symbol,Set{Int}}

    function GenericTensorNetwork()
        new([], Dict(), Dict())
    end

    function GenericTensorNetwork(tensors)
        ind_size = Dict{Symbol,Int}()
        ind_map = Dict{Symbol,Set{Int}}()
        tn = new(NamedDimsArray[], ind_size, ind_map)

        for tensor in tensors
            push!(tn, tensor)
        end

        return tn
    end
end

tensors(tn::GenericTensorNetwork) = tn.tensors
tensors(tn::GenericTensorNetwork, i) = tn.tensors[i]

inds(tn::GenericTensorNetwork) = keys(tn.ind_size)

hyperinds(tn::GenericTensorNetwork) = filter(ind -> count(∋(ind) ∘ dimnames, values(tensors(tn))) > 2, inds(tn))

function Base.push!(tn::GenericTensorNetwork, tensor::NamedDimsArray)
    i = maximum(Iterators.flatten(collect.(values(tn.ind_map))), init = 0) + 1

    for ind in dimnames(tensor)
        if ind in keys(tn.ind_map)
            if tn.ind_size[ind] != size(tensor, ind)
                throw(
                    ArgumentError(
                        "size of index $ind in tensor #$i ($(size(tensor,ind))) does not match previous assigment ($(tn.ind_size[ind]))",
                    ),
                )
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

function Base.popat!(tn::GenericTensorNetwork, i::Integer)
    tensor = popat!(tn.tensors, i)

    for ind in dimnames(tensor)
        delete!(tn.ind_map[ind], i)
        if isempty(tn.ind_map[ind])
            delete!(tn.ind_size, size)
        end
    end

    return tensor
end

function Base.deleteat!(tn::GenericTensorNetwork, i::Integer)
    _ = popat!(tn, i)
    return tn
end

function rand(::Type{GenericTensorNetwork}, n::Integer, reg::Integer; kwargs...)
    output, inputs, size_dict = OptimizedEinsum.rand_equation(n, reg, kwargs...)
    tensors = [NamedDimsArray{tuple(input...)}(rand([size_dict[ind] for ind in input]...)) for input in inputs]
    GenericTensorNetwork(tensors)
end
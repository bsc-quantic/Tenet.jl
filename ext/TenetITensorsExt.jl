module TenetITensorsExt

using Tenet
using ITensors: ITensors, ITensor, Index

function symbolize(index::Index)
    tag = string(ITensors.id(index))

    # NOTE ITensors' Index's tag only has space for 16 characters
    return Symbol(length(tag) > 16 ? tag[(end - 16 + 1):end] : tag)
end

function tagize(index::Symbol)
    tag = string(index)

    # NOTE ITensors' Index's tag only has space for 16 characters
    return length(tag) > 16 ? tag[(end - 16 + 1):end] : tag
end

# TODO customize index names
function Base.convert(::Type{Tensor}, tensor::ITensor)
    array = ITensors.array(tensor)
    is = map(symbolize, ITensors.inds(tensor))
    return Tensor(array, is)
end

function Base.convert(::Type{ITensor}, tensor::Tensor; inds=Dict{Symbol,Index}())
    indices = map(Tenet.inds(tensor)) do i
        haskey(inds, i) ? inds[i] : Index(size(tensor, i), tagize(i))
    end
    return ITensor(parent(tensor), indices)
end

Base.convert(::Type{TensorNetwork}, tn::Vector{ITensor}) = TensorNetwork(map(Tensor, tn))

function Base.convert(::Type{Vector{ITensor}}, tn::Tenet.AbstractTensorNetwork; inds=Dict{Symbol,Index}())
    indices = merge(inds, Dict(
        map(Iterators.filter(!Base.Fix1(haskey, inds), Tenet.inds(tn))) do i
            i => Index(size(tn, i), tagize(i))
        end,
    ))
    return map(tensors(tn)) do tensor
        ITensor(parent(tensor), map(i -> indices[i], Tenet.inds(tensor)))
    end
end

end

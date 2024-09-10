module TenetITensorsExt

using Tenet
using ITensors: ITensors, ITensor, Index

function Tenet.Tensor(tensor::ITensor)
    array = ITensors.array(tensor)
    is = Symbol.(ITensors.id.(ITensors.inds(tensor)))
    return Tensor(array, is)
end

function Base.convert(::Type{ITensor}, tensor::Tensor; inds=Dict{Symbol,Index}())
    indices = map(Tenet.inds(tensor)) do i
        haskey(inds, i) ? inds[i] : Index(size(tensor, i), string(i))
    end
    return ITensor(parent(tensor), indices)
end

Tenet.TensorNetwork(tn::Vector{ITensor}) = TensorNetwork(map(Tensor, tn))

function Base.convert(::Type{Vector{ITensor}}, tn::Tenet.AbstractTensorNetwork; inds=Dict{Symbol,Index}())
    indices = merge(inds, Dict(
        map(Iterators.filter(Base.Fix1(haskey, inds), Tenet.inds(tn))) do i
            i => Index(size(tn, i), string(i))
        end,
    ))
    return map(tensors(tn)) do tensor
        ITensor(parent(tensor), map(i -> indices[i], Tenet.inds(tensor)))
    end
end

end

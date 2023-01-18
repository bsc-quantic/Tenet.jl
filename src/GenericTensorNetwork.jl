import Random: rand

"""
    GenericTensorNetwork

A Tensor Network with arbitrary structure.
"""
struct GenericTensorNetwork <: TensorNetwork
    tensors::Vector{Tensor}
    inds::Dict{Symbol,Index}

    function GenericTensorNetwork()
        new(Tensor[], Dict{Symbol,Index}())
    end
end

function GenericTensorNetwork(tensors)
    # NOTE calling `copy` on each tensor, so tensors are unlinked
    tensors = copy.(tensors)
    indices = Dict{Symbol,Index}()
    tn = GenericTensorNetwork()

    foreach(Base.Fix1(push!, tn), tensors)

    return tn
end

tensors(tn::GenericTensorNetwork) = tn.tensors
tensors(tn::GenericTensorNetwork, i::Integer) = tn.tensors[i]
tensors(tn::GenericTensorNetwork, i::Symbol)::Vector{Tensor} = links(tn.inds[i])
tensors(tn::GenericTensorNetwork, i::Index)::Vector{Tensor} = tensors(tn, nameof(i))

const Sequence{T} = Union{AbstractArray{T,1},NTuple{N,T} where {N}}
tensors(tn::GenericTensorNetwork, inds::Sequence{Union{Symbol,Index}}) = ∩([tensors(tn, i) for i in inds]...)

inds(tn::GenericTensorNetwork) = values(tn.inds)

function Base.push!(tn::GenericTensorNetwork, tensor::Tensor)
    push!(tensors(tn), tensor)

    # TODO link indices
    for i in inds(tensor)
        if i ∉ keys(tn.inds)
            tn.inds[i] = Index(i, size(tensor, i))
        end

        link!(tn.inds[i], tensor)
    end

    return tn
end

Base.append!(A::GenericTensorNetwork, B::GenericTensorNetwork) = (foreach(Fix1(push!, A), B.tensors); A)

function Base.popat!(tn::GenericTensorNetwork, i::Integer)
    tensor = popat!(tensors(tn), i)

    # unlink indices
    for i in inds(tensor)
        index = tn.inds[i]
        unlink!(index, tensor)
    end

    return tensor
end

function Base.deleteat!(tn::GenericTensorNetwork, i::Integer)
    _ = popat!(tn, i)
    return tn
end

function rand(::Type{GenericTensorNetwork}, n::Integer, reg::Integer; kwargs...)
    output, inputs, size_dict = OptimizedEinsum.rand_equation(n, reg, kwargs...)
    tensors = [Tensor(rand([size_dict[ind] for ind in input]...), tuple(input...)) for input in inputs]
    GenericTensorNetwork(tensors)
end
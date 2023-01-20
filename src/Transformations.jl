using DeltaArrays

abstract type Transformation end

function transform! end

function transform!(tn::TensorNetwork, transformations::Sequence{Transformation}) end

function transform(tn::TensorNetwork, transformations::Sequence{Transformation})
    tn = copy(tn)
    transform!(tn, transformations)
    return tn
end

struct HyperindConverter <: Transformation end

"""
    Converts hyperindices to COPY tensors.
"""
function transform!(tn::TensorNetwork, ::Type{HyperindConverter})
    for index in hyperinds(tn)
        # unlink index
        tensors = [pop!(tn, tensor) for tensor in links(index)]

        # replace old index
        indices = Symbol[]
        for (i, tensor) in enumerate(tensors)
            label = Symbol("$(nameof(index))$i")
            push!(indices, label)
            inds = replace(labels(tensor), nameof(index) => label)

            tensor = Tensor(parent(tensor), inds; tensor.meta...)
            push!(tn, tensor)
        end

        # insert COPY tensor
        N = length(indices)
        data = DeltaArray{N}(ones(size(index)))
        tensor = Tensor(data, indices; index.meta...)
        push!(tn, tensor)
    end
end

using DeltaArrays

abstract type Transformation end

transform(tn::TensorNetwork, transformations) = (tn = copy(tn); transform!(tn, transformations); return tn)

function transform! end

transform!(tn::TensorNetwork, transformation::Type{<:Transformation}) = transform!(tn, transformation())

function transform!(tn::TensorNetwork, transformations)
    for transformation in transformations
        transform!(tn, transformation)
    end
    return tn
end

struct HyperindConverter <: Transformation end

"""
    Converts hyperindices to COPY tensors.
"""
function transform!(tn::TensorNetwork, ::HyperindConverter)
    for index in labels(tn, :hyper)
        # dimensionality of `index`
        m = size(tn, index)

        # unlink tensors
        tensors = pop!(tn, index)

        # replace hyperindex for new (non-hyper)index
        new_indices = Symbol[]
        for (i, tensor) in enumerate(tensors)
            label = Symbol("$index$i")
            push!(new_indices, label)

            tensor = replace(tensor, index => label)
            push!(tn, tensor)
        end

        # insert COPY tensor
        N = length(new_indices)
        data = DeltaArray{N}(ones(m))
        tensor = Tensor(data, new_indices; dual = index)
        push!(tn, tensor)
    end
end

# TODO column reduction, diagonal reduction, rank simplification, split simplification

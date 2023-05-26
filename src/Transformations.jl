using DeltaArrays
using OMEinsum

abstract type Transformation end

transform(tn::TensorNetwork, transformations) = (tn = copy(tn); transform!(tn, transformations); return tn)

function transform! end

transform!(tn::TensorNetwork, transformation::Type{<:Transformation}; kwargs...) = transform!(tn, transformation(kwargs...))

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
        tensor = Tensor(data, indices; dual = nameof(index), index.meta...)
        push!(tn, tensor)
    end
end

Base.@kwdef struct DiagonalReduction <: Transformation
    atol::Float64 = 1e-12
    skip::Vector{Index} = Symbol[]
end

function transform!(tn::TensorNetwork, config::DiagonalReduction)
    skip_inds = isempty(config.skip) ? openinds(tn) : config.skip
    queue = collect(keys(tn.tensors))

    while !isempty(queue) # loop over all tensors
        idx = pop!(queue)
        tensor = tn.tensors[idx]

        diag_axes = find_diag_axes(parent(tensor), config.atol)

        while !isempty(diag_axes) # loop over all diagonal axes
            (i, j) = pop!(diag_axes)
            ix_i, ix_j = labels(tensor)[i], labels(tensor)[j]

            # do not reduce output indices
            new, old = (ix_j in nameof.(skip_inds)) ? ((ix_i in nameof.(skip_inds)) ? continue : (ix_j, ix_i)) : (ix_i, ix_j)

            # replace old index in the other tensors in the network
            replacements = 0
            for other_idx in setdiff(keys(tn.tensors)   , idx)
                other_tensor = tn.tensors[other_idx]
                if old in labels(other_tensor)
                    new_tensor = replace(other_tensor, old => new)
                    tn.tensors[other_idx] = new_tensor
                    replacements += 1
                end
            end

            repeated_labels = replace(collect(labels(tensor)), old => new)
            removed_label = filter(l -> l != old, labels(tensor))

            # TODO rewrite with `Tensors` when it supports it
            # extract diagonal
            data = EinCode((String.(repeated_labels),),[String.(removed_label)...])(tensor)
            tn.tensors[idx] = Tensor(data, filter(l -> l != old, labels(tensor)))
            delete!(tn.inds, old)

            # if the new index is in skip_inds, we need to add a COPY tensor
            if new ∈ nameof.(skip_inds)
                data = DeltaArray{replacements+2}(ones(size(tensor, new))) # +2 for the new COPY tensor and the old index
                indices = [Symbol("$(new)$i") for i in 1:replacements+2]
                copy_tensor = Tensor(data, indices)

                # replace the new index in the other tensors in the network
                counter = 1
                for (i, t) in enumerate(tn.tensors)
                    if new in labels(t)
                        new_tensor = replace(t, new => indices[counter])
                        tn.tensors[i] = new_tensor
                        counter += 1
                    end
                end

                push!(tn, copy_tensor)
            end

            tensor = tn.tensors[idx]
            diag_axes = find_diag_axes(parent(tensor), config.atol)
        end
    end

    return tn
end

function find_diag_axes(x::AbstractArray, atol=1e-12)
    ndims = size(x)

    # Find all the potential diagonals
    potential_diag_axes = [(i, j) for i in 1:length(ndims) for j in i+1:length(ndims) if ndims[i] == ndims[j]]

    # Check what elements satisfy the condition
    return filter(potential_diag_axes) do (d1, d2)
        all(pairs(x)) do (idx, val)
            idx[d1] == idx[d2] || abs(val) <= atol
        end
    end
end

# TODO column reduction, rank simplification, split simplification

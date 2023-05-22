using DeltaArrays

abstract type Transformation end

transform(tn::TensorNetwork, transformations; kwargs...) = (tn = copy(tn); transform!(tn, transformations; kwargs...); return tn)

function transform! end

transform!(tn::TensorNetwork, transformation::Type{<:Transformation}; kwargs...) = transform!(tn, transformation(); kwargs...)

function transform!(tn::TensorNetwork, transformations; kwargs...)
    for transformation in transformations
        transform!(tn, transformation; kwargs...)
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

struct DiagonalReduction <: Transformation end

function transform!(tn::TensorNetwork, ::DiagonalReduction; output_inds=nothing, atol=1e-12)
    output_inds = output_inds === nothing ? openinds(tn) : output_inds
    queue = collect(keys(tn.tensors))
    cache = Dict()

    while !isempty(queue)
        idx = pop!(queue)
        tensor = tn.tensors[idx]
        cache_key = ("dr", idx, objectid(tensor.data))

        if haskey(cache, cache_key)
            continue
        end

        diag_axes = find_diag_axes(tensor.data, atol)

        if isempty(diag_axes)
            cache[cache_key] = true
            continue
        end

        for (i, j) in diag_axes
            ix_i, ix_j = labels(tensor)[i], labels(tensor)[j]
            new, old = (ix_j in output_inds) ? ((ix_i in output_inds) ? continue : (ix_j, ix_i)) : (ix_i, ix_j)

            for other_idx in setdiff(keys(tn.tensors), idx)
                other_tensor = tn.tensors[other_idx]
                if old in labels(other_tensor)
                    new_tensor = replace(other_tensor, old => new)
                    tn.tensors[other_idx] = new_tensor
                end
            end

            data = view(tensor, old => 1)
            tn.tensors[idx] = Tensor(data, filter(l -> l != old, labels(tensor)))

            !isempty(find_diag_axes(parent(tensor), atol)) && push!(queue, idx)
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

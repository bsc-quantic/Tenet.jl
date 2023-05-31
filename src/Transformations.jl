using DeltaArrays
using EinExprs
using OMEinsum
using UUIDs: uuid4

abstract type Transformation end

transform(tn::TensorNetwork, transformations) = (tn = deepcopy(tn); transform!(tn, transformations); return tn)

function transform! end

transform!(tn::TensorNetwork, transformation::Type{<:Transformation}; kwargs...) =
    transform!(tn, transformation(kwargs...))

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

        diag_axes = find_diag_axes(parent(tensor), atol = config.atol)

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
            data = EinCode((String.(repeated_labels),), [String.(removed_label)...])(tensor)
            tn.tensors[idx] = Tensor(data, filter(l -> l != old, labels(tensor)))
            delete!(tn.inds, old)

            # if the new index is in skip_inds, we need to add a COPY tensor
            if new ∈ nameof.(skip_inds)
                data = DeltaArray{replacements+2}(ones(size(tensor, new))) # +2 for the new COPY tensor and the old index
                indices = [Symbol(uuid4()) for i in 1:replacements+1]
                copy_tensor = Tensor(data, (new, indices...))

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
            diag_axes = find_diag_axes(parent(tensor), atol = config.atol)
        end
    end

    return tn
end

struct RankSimplification <: Transformation end

function transform!(tn::TensorNetwork, ::RankSimplification)
    @label rank_transformation_start
    for tensor in tensors(tn)
        # TODO replace this code for `neighbours` method
        connected_tensors = mapreduce(label -> select(tn, label), ∪, labels(tensor))
        filter!(!=(tensor), connected_tensors)

        for c_tensor in connected_tensors
            path = EinExpr([tensor, c_tensor])

            # Check if contraction does not increase the rank
            # TODO implement `removedrank` counter on EinExprs and let it choose function
            if ndims(path) <= maximum(ndims.(path.args))
                new_tensor = contract(path)

                # Update tensor network
                push!(tn, new_tensor)
                delete!(tn, tensor)
                delete!(tn, c_tensor)

                # Break the loop since we modified the network and need to recheck connections
                @goto rank_transformation_start
            end
        end
    end

    return tn
end

Base.@kwdef struct AntiDiagonalGauging <: Transformation
    atol::Float64 = 1e-12
    skip::Vector{Index} = Symbol[]
end

function transform!(tn::TensorNetwork, config::AntiDiagonalGauging)
    skip_inds = isempty(config.skip) ? openinds(tn) : config.skip

    for idx in keys(tn.tensors)
        tensor = tn.tensors[idx]

        anti_diag_axes = find_anti_diag_axes(parent(tensor), atol = config.atol)

        for (i, j) in anti_diag_axes # loop over all anti-diagonal axes
            ix_i, ix_j = labels(tensor)[i], labels(tensor)[j]

            # do not gauge output indices
            _, ix_to_gauge =
                (ix_j ∈ nameof.(skip_inds)) ? ((ix_i ∈ nameof.(skip_inds)) ? continue : (ix_j, ix_i)) : (ix_i, ix_j)

            # reverse the order of ix_to_gauge in all tensors where it appears
            for t in tensors(tn)
                ix_to_gauge in labels(t) && reverse!(parent(t), dims = findfirst(l -> l == ix_to_gauge, labels(t)))
            end
        end
    end

    return tn
end

Base.@kwdef struct ColumnReduction <: Transformation
    atol::Float64 = 1e-12
    skip::Vector{Index} = Symbol[]
end

function transform!(tn::TensorNetwork, config::ColumnReduction)
    skip_inds = isempty(config.skip) ? openinds(tn) : config.skip

    for idx in keys(tn.tensors)
        tensor = tn.tensors[idx]

        column_axes = find_column_axes(parent(tensor), atol = config.atol)

        for (d, c) in column_axes # loop over all column axes
            ix_i = labels(tensor)[d]

            # do not reduce output indices
            if ix_i ∈ nameof.(skip_inds)
                continue
            end

            # reduce all tensors where ix_i appears
            for (ind, t) in enumerate(tensors(tn))
                if ix_i ∈ labels(t)
                    reduced_dims = [i == ix_i ? filter(j -> j != c, 1:size(t, i)) : (1:size(t, i)) for i in labels(t)]
                    tn.tensors[ind] = Tensor(view(parent(t), reduced_dims...), labels(t))
                end
            end
        end
    end

    return tn
end

function find_column_axes(x; atol=1e-12)
    ndims = size(x)

    column_axes = []
    for d in 1:length(ndims)
        for i in 1:ndims[d]
            column = selectdim(x, d, i)
            if all(abs.(column) .<= atol)
                push!(column_axes, (d, i))
            end
        end
    end
    return unique(column_axes)
end

function find_diag_axes(x; atol = 1e-12)
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

function find_anti_diag_axes(x; atol = 1e-12)
    ndims = size(x)

    # Find all the potential anti-diagonals
    potential_anti_diag_axes = [(i, j) for i in 1:length(ndims) for j in i+1:length(ndims) if ndims[i] == ndims[j]]

    # Check what elements satisfy the condition
    return filter(potential_anti_diag_axes) do (d1, d2)
        d = ndims[d1] # Since d1 and d2 are the same size
        all(pairs(x)) do (idx, val)
            idx[d1] != d - idx[d2] || abs(val) <= atol
        end
    end
end

# TODO split simplification
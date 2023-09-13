using DeltaArrays
using EinExprs
using OMEinsum
using UUIDs: uuid4
using Tenet: parenttype
using Combinatorics: combinations

abstract type Transformation end

"""
    transform(tn::TensorNetwork, config::Transformation)
    transform(tn::TensorNetwork, configs)

Return a new [`TensorNetwork`](@ref) where some `Transformation` has been performed into it.

See also: [`transform!`](@ref).
"""
transform(tn::TensorNetwork, transformations) = (tn = deepcopy(tn); transform!(tn, transformations); return tn)

"""
    transform!(tn::TensorNetwork, config::Transformation)
    transform!(tn::TensorNetwork, configs)

In-place version of [`transform`](@ref).
"""
function transform! end

transform!(tn::TensorNetwork, transformation::Type{<:Transformation}; kwargs...) =
    transform!(tn, transformation(kwargs...))

function transform!(tn::TensorNetwork, transformations)
    for transformation in transformations
        transform!(tn, transformation)
    end
    return tn
end

"""
    HyperindConverter <: Transformation

Convert hyperindices to COPY-tensors, represented by `DeltaArray`s.
"""
struct HyperindConverter <: Transformation end

function transform!(tn::TensorNetwork, ::HyperindConverter)
    for index in inds(tn, :hyper)
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

"""
    DiagonalReduction <: Transformation

Reduce the dimension of a `Tensor` in a [`TensorNetwork`](@ref) when it has a pair of indices that fulfil a diagonal structure.

# Keyword Arguments

  - `atol` Absolute tolerance. Defaults to `1e-12`.
"""
Base.@kwdef struct DiagonalReduction <: Transformation
    atol::Float64 = 1e-12
end

function transform!(tn::TensorNetwork, config::DiagonalReduction)
    for tensor in filter(tensor -> !(parenttype(typeof(tensor)) <: DeltaArray), tensors(tn))
        diaginds = find_diag_axes(tensor, atol = config.atol)
        isempty(diaginds) && continue

        transformed_tensor = reduce(diaginds; init = (; target = tensor, copies = Tensor[])) do (target, copies), inds
            N = length(inds)

            # insert COPY tensor
            new_index = Symbol(uuid4())
            data = DeltaArray{N + 1}(ones(size(target, first(inds))))
            push!(copies, Tensor(data, (new_index, inds...), dual = new_index))

            # extract diagonal of target tensor
            # TODO rewrite using `einsum!` when implemented in Tensors
            data = EinCode(
                (String.(replace(Tenet.inds(target), [i => first(inds) for i in inds[2:end]]...)),),
                String.(filter(∉(inds[2:end]), Tenet.inds(target))),
            )(
                target,
            )
            target = Tensor(
                data,
                map(index -> index === first(inds) ? new_index : index, filter(∉(inds[2:end]), Tenet.inds(target)));
                target.meta...,
            )

            return (; target = target, copies = copies)
        end

        transformed_tn = TensorNetwork([transformed_tensor.target, transformed_tensor.copies...])
        replace!(tn, tensor => transformed_tn)
    end

    return tn
end

"""
    RankSimplification <: Transformation

Preemptively contract tensors whose result doesn't increase in size.
"""
struct RankSimplification <: Transformation end

function transform!(tn::TensorNetwork, ::RankSimplification)
    @label rank_transformation_start
    for tensor in tensors(tn)
        # TODO replace this code for `neighbours` method
        connected_tensors = mapreduce(label -> select(tn, label), ∪, inds(tensor))
        filter!(!=(tensor), connected_tensors)

        for c_tensor in connected_tensors
            # TODO keep output inds?
            path = sum([
                EinExpr(inds(tensor), Dict(index => size(tensor, index) for index in inds(tensor))) for
                tensor in [tensor, c_tensor]
            ])

            # Check if contraction does not increase the rank
            EinExprs.removedrank(path) < 0 && continue

            new_tensor = contract(tensor, c_tensor)

            # Update tensor network
            push!(tn, new_tensor)
            delete!(tn, tensor)
            delete!(tn, c_tensor)

            # Break the loop since we modified the network and need to recheck connections
            @goto rank_transformation_start
        end
    end

    return tn
end

"""
    AntiDiagonalGauging <: Transformation

Reverse the order of tensor indices that fulfill the anti-diagonal condition.
While this transformation doesn't directly enhance computational efficiency, it sets up the [`TensorNetwork`](@ref) for other operations that do.

# Keyword Arguments

  - `atol` Absolute tolerance. Defaults to `1e-12`.
  - `skip` List of indices to skip. Defaults to `[]`.
"""
Base.@kwdef struct AntiDiagonalGauging <: Transformation
    atol::Float64 = 1e-12
    skip::Vector{Symbol} = Symbol[]
end

function transform!(tn::TensorNetwork, config::AntiDiagonalGauging)
    skip_inds = isempty(config.skip) ? inds(tn, set = :open) : config.skip

    for idx in keys(tn.tensors)
        tensor = tn.tensors[idx]

        anti_diag_axes = find_anti_diag_axes(parent(tensor), atol = config.atol)

        for (i, j) in anti_diag_axes # loop over all anti-diagonal axes
            ix_i, ix_j = inds(tensor)[i], inds(tensor)[j]

            # do not gauge output indices
            _, ix_to_gauge = (ix_j ∈ skip_inds) ? ((ix_i ∈ skip_inds) ? continue : (ix_j, ix_i)) : (ix_i, ix_j)

            # reverse the order of ix_to_gauge in all tensors where it appears
            for t in tensors(tn)
                ix_to_gauge in inds(t) && reverse!(parent(t), dims = findfirst(l -> l == ix_to_gauge, inds(t)))
            end
        end
    end

    return tn
end

"""
    ColumnReduction <: Transformation

Truncate the dimension of a `Tensor` in a [`TensorNetwork`](@ref) when it contains columns with all elements smaller than `atol`.

# Keyword Arguments

  - `atol` Absolute tolerance. Defaults to `1e-12`.
  - `skip` List of indices to skip. Defaults to `[]`.
"""
Base.@kwdef struct ColumnReduction <: Transformation
    atol::Float64 = 1e-12
    skip::Vector{Symbol} = Symbol[]
end

function transform!(tn::TensorNetwork, config::ColumnReduction)
    skip_inds = isempty(config.skip) ? inds(tn, set = :open) : config.skip

    for tensor in tn.tensors
        zero_columns = find_zero_columns(parent(tensor), atol = config.atol)
        zero_columns_by_axis = [filter(x -> x[1] == d, zero_columns) for d in 1:length(size(tensor))]

        # find non-zero column for each axis
        non_zero_columns =
            [(d, setdiff(1:size(tensor, d), [x[2] for x in zero_columns_by_axis[d]])) for d in 1:length(size(tensor))]

        # remove axes that have more than one non-zero column
        axes_to_reduce = [(d, c[1]) for (d, c) in filter(x -> length(x[2]) == 1, non_zero_columns)]

        # First try to reduce the whole index if only one column is non-zeros
        for (d, c) in axes_to_reduce # loop over all column axes
            ix_i = inds(tensor)[d]

            # do not reduce output indices
            if ix_i ∈ skip_inds
                continue
            end

            # reduce all tensors where ix_i appears
            for (ind, t) in enumerate(tensors(tn))
                if ix_i ∈ inds(t)
                    # Replace the tensor with the reduced one
                    new_tensor = selectdim(parent(t), findfirst(l -> l == ix_i, inds(t)), c)
                    new_inds = filter(l -> l != ix_i, inds(t))

                    tn.tensors[ind] = Tensor(new_tensor, new_inds)
                end
            end
            delete!(tn.indices, ix_i)
        end

        # Then try to reduce the dimensionality of the index in the other tensors
        zero_columns = find_zero_columns(parent(tensor), atol = config.atol)
        for (d, c) in zero_columns # loop over all column axes
            ix_i = inds(tensor)[d]

            # do not reduce output indices
            if ix_i ∈ skip_inds
                continue
            end

            # reduce all tensors where ix_i appears
            for (ind, t) in enumerate(tensors(tn))
                if ix_i ∈ inds(t)
                    reduced_dims = [i == ix_i ? filter(j -> j != c, 1:size(t, i)) : (1:size(t, i)) for i in inds(t)]
                    tn.tensors[ind] = Tensor(view(parent(t), reduced_dims...), inds(t))
                end
            end
        end
    end

    return tn
end

"""
    SplitSimplification <: Transformation

Reduce the rank of tensors in the [`TensorNetwork`](@ref) by decomposing them using the Singular Value Decomposition (SVD). Tensors whose factorization do not increase the maximum rank of the network are left decomposed.

# Keyword Arguments

  - `atol` Absolute tolerance. Defaults to `1e-10`.
"""
Base.@kwdef struct SplitSimplification <: Transformation
    atol::Float64 = 1e-10  # A threshold for SVD rank determination
end

function transform!(tn::TensorNetwork, config::SplitSimplification)
    @label split_simplification_start
    for tensor in tensors(tn)
        inds = Tenet.inds(tensor)

        # iterate all bipartitions of the tensor's indices
        bipartitions = Iterators.flatten(combinations(inds, r) for r in 1:(length(inds)-1))
        for bipartition in bipartitions
            left_inds = collect(bipartition)
            right_inds = setdiff(inds, left_inds)

            # perform an SVD across the bipartition
            u, s, v = svd(tensor; left_inds = left_inds)
            rank_s = sum(diag(s) .> config.atol)

            if rank_s < size(s, 1)
                # truncate data
                u = view(u, Tenet.inds(s)[1] => 1:rank_s)
                s = view(s, (idx -> idx => 1:rank_s).(Tenet.inds(s))...)
                v = view(v, Tenet.inds(s)[2] => 1:rank_s)

                # replace the original tensor with factorization
                tensor_l = u * s
                tensor_r = v

                push!(tn, dropdims(tensor_l))
                push!(tn, dropdims(tensor_r))
                pop!(tn, tensor)

                # iterator is no longer valid, so restart loop
                @goto split_simplification_start
            end
        end
    end
    return tn
end

function find_zero_columns(x; atol = 1e-12)
    dims = size(x)

    # Create an initial set of all possible column pairs
    zero_columns = Set((d, c) for d in 1:length(dims) for c in 1:dims[d])

    # Iterate over each element in tensor
    for index in CartesianIndices(x)
        val = x[index]

        # For each non-zero element, eliminate the corresponding column from the zero_columns set
        if abs(val) > atol
            for d in 1:length(dims)
                c = index[d]
                delete!(zero_columns, (d, c))
            end
        end
    end

    # Now the zero_columns set only contains column pairs where all elements are zero
    return collect(zero_columns)
end

function find_diag_axes(x; atol = 1e-12)
    if ndims(parent(x)) == 1 # Skip 1D tensors
        return []
    end

    # find all the potential diagonals
    potential_diag_axes = [(i, j) for i in 1:ndims(x) for j in i+1:ndims(x) if size(x, i) == size(x, j)]

    # check what elements satisfy the condition
    diag_pairs = filter(potential_diag_axes) do (d1, d2)
        all(pairs(parent(x))) do (idx, val)
            idx[d1] == idx[d2] || abs(val) <= atol
        end
    end

    # if overlap between pairs of diagonal axes, then all involved axes are diagonal
    diag_sets = reduce(diag_pairs; init = Vector{Int}[]) do acc, pair
        i = findfirst(set -> !isdisjoint(set, pair), acc)
        !isnothing(i) ? union!(acc[i], pair) : push!(acc, collect(pair))
        return acc
    end

    # map to index symbols
    map(set -> map(i -> inds(x)[i], set), diag_sets)
end

function find_anti_diag_axes(x; atol = 1e-12)
    if ndims(parent(x)) == 1 # Skip 1D tensors
        return []
    end

    # Find all the potential anti-diagonals
    potential_anti_diag_axes = [(i, j) for i in 1:ndims(x) for j in i+1:ndims(x) if size(x, i) == size(x, j)]

    # Check what elements satisfy the condition
    return filter(potential_anti_diag_axes) do (d1, d2)
        all(pairs(parent(x))) do (idx, val)
            idx[d1] != size(x, d1) - idx[d2] || abs(val) <= atol
        end
    end
end

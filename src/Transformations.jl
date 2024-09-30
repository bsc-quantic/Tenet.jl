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

function transform!(tn::TensorNetwork, transformation::Type{<:Transformation}; kwargs...)
    return transform!(tn, transformation(kwargs...))
end

function transform!(tn::TensorNetwork, transformations)
    for transformation in transformations
        transform!(tn, transformation)
    end
    return tn
end

"""
    HyperFlatten <: Transformation

Convert hyperindices to COPY-tensors, represented by `DeltaArray`s.
This transformation is always used by default when visualizing a `TensorNetwork` with `plot`.

See also: [`HyperGroup`](@ref).
"""
struct HyperFlatten <: Transformation end

function hyperflatten(tn::AbstractTensorNetwork)
    return Dict(
        map(inds(tn; set=:hyper)) do hyperindex
            n = length(tensors(tn; intersects=hyperindex))
            map(1:n) do i
                Symbol("$hyperindex$i")
            end => hyperindex
        end,
    )
end

function transform!(tn::TensorNetwork, ::HyperFlatten)
    for (flatindices, hyperindex) in hyperflatten(tn)
        # insert COPY tensor
        array = DeltaArray{length(flatindices)}(ones(size(tn, hyperindex)))
        tensor = Tensor(array, flatindices)
        push!(tn, tensor)

        # replace hyperindex for new flat Indices
        # TODO move this part to `replace!`?
        tensors = pop!(tn, hyperindex)
        for (flatindex, tensor) in zip(flatindices, tensors)
            tensor = replace(tensor, hyperindex => flatindex)
            push!(tn, tensor)
        end
    end
end

"""
    HyperGroup <: Transformation

Convert COPY-tensors, represented by `DeltaArray`s, to hyperindices.

See also: [`HyperFlatten`](@ref).
"""
struct HyperGroup <: Transformation end

function transform!(tn::TensorNetwork, ::HyperGroup)
    targets = Iterators.filter(x -> parenttype(x) <: DeltaArray, tensors(tn))

    open_indices = inds(tn; set=:open)
    targets = Iterators.filter(t -> isdisjoint(inds(t), open_indices), targets)

    for tensor in targets
        # remove COPY tensor
        delete!(tn, tensor)

        # insert hyperindex
        hyperindex = Symbol(uuid4())

        # insert weights vector
        if !all(isone, delta(parent(tensor)))
            push!(tn, Tensor(delta(parent(tensor)), [hyperindex]))
        end

        for flatindex in inds(tensor)
            tensor = pop!(tn, only(tensors(tn; contains=flatindex)))
            tensor = replace(tensor, flatindex => hyperindex)
            push!(tn, tensor)
        end
    end

    return tn
end

"""
    ContractSimplification <: Transformation

Preemptively contract tensors whose result doesn't increase in size.
"""
@kwdef struct ContractSimplification <: Transformation
    minimize::Symbol = :length
    recursive::Bool = true

    function ContractSimplification(minimize::Symbol, recursive::Bool=true)
        @assert minimize in (:length, :rank)
        return new(minimize, recursive)
    end
end

function transform!(tn::TensorNetwork, config::ContractSimplification)
    targets = inds(tn; set=:inner)

    for index in targets
        # check if the index is already contracted
        index ∈ inds(tn; set=:inner) || continue

        # check that the contraction minimizes the size/rank
        candidate_tensors = tensors(tn; contains=index)
        result = sum([
            EinExpr(inds(tensor), Dict(index => size(tensor, index) for index in inds(tensor))) for
            tensor in candidate_tensors
        ])

        winner = if config.minimize == :rank
            ndims(result) <= maximum(ndims, candidate_tensors)
        else # :length
            removedsize(result) >= 0
        end

        winner && contract!(tn, index)
    end

    # if the network has been modified, recursively apply the transformation
    if length(inds(tn; set=:inner)) < length(targets) && config.recursive
        transform!(tn, config)
    end

    return tn
end

"""
    Truncate <: Transformation

Truncate the dimension of a `Tensor` in a [`TensorNetwork`](@ref) when it contains columns with all elements smaller than `atol`.

# Keyword Arguments

  - `atol` Absolute tolerance. Defaults to `1e-12`.
  - `skip` List of indices to skip. Defaults to `[]`.
"""
Base.@kwdef struct Truncate <: Transformation
    atol::Float64 = 1e-12
    skip::Vector{Symbol} = Symbol[]
end

function transform!(tn::TensorNetwork, config::Truncate)
    skip_inds = isempty(config.skip) ? inds(tn; set=:open) : config.skip

    for tensor in tensors(tn)
        for (dim, index) in enumerate(inds(tensor))
            index ∈ skip_inds && continue

            zeroslices = iszero.(eachslice(tensor; dims=dim))
            any(zeroslices) || continue

            slice!(tn, index, count(!, zeroslices) == 1 ? findfirst(!, zeroslices) : findall(!, zeroslices))
        end
    end

    return tn
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
        diaginds = find_diag_axes(tensor; atol=config.atol)
        isempty(diaginds) && continue

        transformed_tensor = reduce(diaginds; init=(; target=tensor, copies=Tensor[])) do (target, copies), inds
            N = length(inds)

            # insert COPY tensor
            new_index = Symbol(uuid4())
            data = DeltaArray{N + 1}(ones(size(target, first(inds))))
            push!(copies, Tensor(data, (new_index, inds...)))

            # extract diagonal of target tensor
            # TODO rewrite using `einsum!` when implemented in Tensors
            data = EinCode(
                (String.(replace(Tenet.inds(target), [i => first(inds) for i in inds[2:end]]...)),),
                String.(filter(∉(inds[2:end]), Tenet.inds(target))),
            )(
                target
            )
            target = Tensor(
                data,
                map(index -> index === first(inds) ? new_index : index, filter(∉(inds[2:end]), Tenet.inds(target)));
            )

            return (; target=target, copies=copies)
        end

        transformed_tn = TensorNetwork(Tensor[transformed_tensor.target, transformed_tensor.copies...])
        replace!(tn, tensor => transformed_tn)
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
    skip_inds = isempty(config.skip) ? inds(tn; set=:open) : config.skip

    for tensor in keys(tn.tensormap)
        anti_diag_axes = find_anti_diag_axes(parent(tensor); atol=config.atol)

        for (i, j) in anti_diag_axes # loop over all anti-diagonal axes
            ix_i, ix_j = inds(tensor)[i], inds(tensor)[j]

            # do not gauge output indices
            _, ix_to_gauge = (ix_j ∈ skip_inds) ? ((ix_i ∈ skip_inds) ? continue : (ix_j, ix_i)) : (ix_i, ix_j)

            # reverse the order of ix_to_gauge in all tensors where it appears
            for t in tensors(tn)
                ix_to_gauge in inds(t) && reverse!(parent(t); dims=findfirst(l -> l == ix_to_gauge, inds(t)))
            end
        end
    end

    return tn
end

"""
    SplitSimplification <: Transformation

Reduce the rank of tensors in the [`TensorNetwork`](@ref) by decomposing them using the Singular Value Decomposition (SVD).
Tensors whose factorization do not increase the maximum rank of the network are left decomposed.

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
        bipartitions = Iterators.flatten(combinations(inds, r) for r in 1:(length(inds) - 1))
        for bipartition in bipartitions
            left_inds = collect(bipartition)

            # perform an SVD across the bipartition
            u, s, v = svd(tensor; left_inds=left_inds)
            rank_s = sum(s .> config.atol)

            if rank_s < length(s)
                hyperindex = only(Tenet.inds(s))

                # truncate data
                u = view(u, hyperindex => 1:rank_s)
                s = view(s, hyperindex => 1:rank_s)
                v = view(v, hyperindex => 1:rank_s)

                # replace the original tensor with factorization
                tensor_l = contract(u, s; dims=Symbol[])
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

function find_diag_axes(x; atol=1e-12)
    # skip 1D tensors
    ndims(parent(x)) == 1 && return []

    # find all the potential diagonals
    potential_diag_axes = [(i, j) for i in 1:ndims(x) for j in (i + 1):ndims(x) if size(x, i) == size(x, j)]

    # check what elements satisfy the condition
    diag_pairs = filter(potential_diag_axes) do (d1, d2)
        all(pairs(parent(x))) do (idx, val)
            idx[d1] == idx[d2] || abs(val) <= atol
        end
    end

    # if overlap between pairs of diagonal axes, then all involved axes are diagonal
    diag_sets = reduce(diag_pairs; init=Vector{Int}[]) do acc, pair
        i = findfirst(set -> !isdisjoint(set, pair), acc)
        !isnothing(i) ? union!(acc[i], pair) : push!(acc, collect(pair))
        return acc
    end

    # map to index symbols
    return map(set -> map(i -> inds(x)[i], set), diag_sets)
end

function find_anti_diag_axes(x; atol=1e-12)
    # skip 1D tensors
    ndims(parent(x)) == 1 && return []

    # Find all the potential anti-diagonals
    potential_anti_diag_axes = [(i, j) for i in 1:ndims(x) for j in (i + 1):ndims(x) if size(x, i) == size(x, j)]

    # Check what elements satisfy the condition
    return filter(potential_anti_diag_axes) do (d1, d2)
        all(pairs(parent(x))) do (idx, val)
            idx[d1] != size(x, d1) - idx[d2] || abs(val) <= atol
        end
    end
end

module TenetDaggerExt

using Tenet
using Dagger: Dagger, ArrayOp, Context, ArrayDomain

struct Contract{T,N} <: ArrayOp{T,N}
    ic::Vector{Symbol}
    a::ArrayOp
    ia::Vector{Symbol}
    b::ArrayOp
    ib::Vector{Symbol}

    function Contract(ic, a, ia, b, ib)
        allunique(ia) || throw(ErrorException("ia must have unique indices"))
        allunique(ib) || throw(ErrorException("ib must have unique indices"))
        allunique(ic) || throw(ErrorException("ic must have unique indices"))
        ic ⊆ ia ∪ ib || throw(ErrorException("ic must be a subset of ia ∪ ib"))
        return new{promote_type(eltype(a), eltype(b)),length(ic)}(ic, a, ia, b, ib)
    end
end

function Base.size(x::Contract)
    return Tuple(
        Iterators.map(x.ic) do i
            if i ∈ x.ia
                size(x.a, findfirst(==(i), x.ia))
            elseif i ∈ x.ib
                size(x.b, findfirst(==(i), x.ib))
            else
                throw(ErrorException("index $i not found in a nor b"))
            end
        end,
    )
end

function Dagger.Blocks(x::Contract)
    return Dagger.Blocks(map(x.ic) do i
        j = findfirst(==(i), x.ia)
        isnothing(j) || return x.a.partitioning[j]

        j = findfirst(==(i), x.ia)
        isnothing(j) || return x.b.partitioning[j]

        throw(ErrorException("index $i not found in a nor b"))
    end...)
end

function selectdims(a, proj::Pair...)
    reduce(proj; init=a) do acc, (d, i)
        selectdim(acc, d, i)
    end
end

contractfn(ic, chunk_a, ia, chunk_b, ib) = parent(contract(Tensor(chunk_a, ia), Tensor(chunk_b, ib); out=ic))

function Dagger.stage(ctx::Context, op::Contract{T,N}) where {T,N}
    domain = Dagger.ArrayDomain([1:l for l in size(op)])
    partitioning = Dagger.Blocks(op)

    # NOTE careful with ÷ for dividing into partitions
    subdomains = Array{ArrayDomain{N}}(undef, map(÷, size(op), partitioning.blocksize))
    for indices in eachindex(IndexCartesian(), subdomains)
        subdomains[indices...] = ArrayDomain(
            map(indices, partitioning.blocksize) do i, step
                (i - 1) * step .+ (1:step)
            end,
        )
    end

    suminds = setdiff(op.ia ∪ op.ib, op.ic)
    inner_perm_a = map(i -> findfirst(==(i), op.ia), suminds)
    inner_perm_b = map(i -> findfirst(==(i), op.ib), suminds)

    mask_a = op.ic .∈ op.ia
    mask_b = op.ic .∈ op.ib
    outer_perm_a = map(i -> findfirst(==(i), op.ia), op.ic[mask_a])
    outer_perm_b = map(i -> findfirst(==(i), op.ib), op.ic[mask_b])

    chunks = similar(subdomains, EagerThunk)
    for indices in eachindex(IndexCartesian(), chunks)
        outer_indices_a = indices[mask_a]
        chunks_a = reduce(zip(outer_perm_a, outer_indices_a); init=Dagger.chunks(op.a)) do acc, (d, i)
            selectdim(acc, d, i:i)
        end
        chunks_a = permutedims(chunks_a, inner_perm_a)

        outer_indices_b = indices[mask_b]
        chunks_b = reduce(zip(outer_perm_b, outer_indices_b); init=Dagger.chunks(op.b)) do acc, (d, i)
            selectdim(acc, d, i:i)
        end
        chunks_b = permutedims(chunks_b, inner_perm_b)

        chunks[indices...] = Dagger.treereduce(
            Dagger.AddComputeOp,
            map(chunks_a, chunks_b) do chunk_a, chunk_b
                # TODO add ThunkOptions: alloc_util, occupancy, ...
                Dagger.@spawn contractfn(op.ic, chunk_a, op.ia, chunk_b, op.ib)
            end,
        )
    end

    return DArray(T, domain, subdomains, chunks, partitioning)
end

function Tenet.contract(
    a::Tensor{Ta,Na,Aa}, b::Tensor{Tb,Nb,Ab}; dims=(∩(inds(a), inds(b))), out=nothing
) where {Ta,Tb,Na,Nb,Aa<:Dagger.DArray{Ta,Na},Ab<:Dagger.DArray{Tb,Nb}}
    ia = collect(inds(a))
    ib = collect(inds(b))
    i = ∩(dims, ia, ib)

    ic::Vector{Symbol} = if isnothing(out)
        setdiff(ia ∪ ib, i isa Base.AbstractVecOrTuple ? i : (i,))::Vector{Symbol}
    else
        out
    end

    data = Dagger._to_darray(Contract(ic, parent(a), ia, parent(b), ib))
    return Tensor(data, ic)
end

Tenet.contract(a::Tensor, b::Tensor{T,N,A}; kwargs...) where {T,N,A<:Dagger.DArray} = contract(b, a; kwargs...)
function Tenet.contract(a::Tensor{T,N,A}, b::Tensor; kwargs...) where {T,N,A<:Dagger.DArray}
    throw(
        ArgumentError("contract on a Dagger.DArray-backed Tensor with a non-DArray-backed Tensor is not yet supported")
    )
end

end

module TenetDaggerExt

using Tenet
using Dagger: Dagger, ArrayOp, Context, ArrayDomain

struct Contract{T,N} <: ArrayOp{T,N}
    a::ArrayOp
    ia::Vector{Symbol}
    b::ArrayOp
    ib::Vector{Symbol}
    ic::Vector{Symbol}

    function Contract(a, ia, b, ib, ic)
        allunique(ia) || throw(ErrorException("ia must have unique indices"))
        allunique(ib) || throw(ErrorException("ib must have unique indices"))
        allunique(ic) || throw(ErrorException("ic must have unique indices"))
        new{promote_type(eltype(a), eltype(b)),length(ic)}(a, ia, b, ib, ic)
    end
end

Base.size(x::Contract) = map(x.ic) do i
    i ∈ x.ia && return size(x.a, findfirst(==(i), x.ia))
    i ∈ x.ib && return size(x.b, findfirst(==(i), x.ib))
    throw(ErrorException("index $i not found in a nor b"))
end |> splat(tuple)

Dagger.Blocks(x::Contract) = map(x.ic) do i
    j = findfirst(==(i), x.ia)
    isnothing(j) || return x.a.partitioning.blocksize[j]

    j = findfirst(==(i), x.ib)
    isnothing(j) || return x.b.partitioning.blocksize[j]

    throw(ErrorException("index $i not found in a nor b"))
end |> splat(Dagger.Blocks)

selectdims(a, proj::Pair...) =
    reduce(proj; init = a) do acc, (d, i)
        selectdim(acc, d, i)
    end

function Dagger.stage(ctx::Context, op::Contract{T,N}) where {T,N}
    domain = Dagger.ArrayDomain([1:l for l in size(op)])
    partitioning = Dagger.Blocks(op)

    subdoms_start = tuple([1 for _ in 1:length(partitioning.blocksize)]...)
    subdoms_cumlength = tuple([collect(dim:2) for dim in partitioning.blocksize]...)
    subdomains = Dagger.DomainBlocks{length(partitioning.blocksize)}(subdoms_start, subdoms_cumlength)

    contractor =
        EinCode((Tenet.__omeinsum_sym2str(op.ia), Tenet.__omeinsum_sym2str(op.ib)), Tenet.__omeinsum_sym2str(op.ic))

    suminds = setdiff(op.ia ∪ op.ib, op.ic)
    inner_perm_a = map(i -> findfirst(==(i), op.ia), suminds)
    inner_perm_b = map(i -> findfirst(==(i), op.ib), suminds)

    mask_a = map(∈(op.ia), op.ic)
    mask_b = map(∈(op.ib), op.ic)
    outer_perm_a = map(i -> findfirst(==(i), op.ia), op.ic[mask_a])
    outer_perm_b = map(i -> findfirst(==(i), op.ib), op.ic[mask_b])

    chunks = similar(subdomains, Dagger.EagerThunk)
    for indices in eachindex(IndexCartesian(), chunks)
        outer_indices_a = Tuple(indices)[mask_a]
        chunks_a = reduce(zip(outer_perm_a, outer_indices_a); init = Dagger.chunks(op.a)) do acc, (d, i)
            selectdim(acc, d, i:i)
        end
        chunks_a = permutedims(chunks_a, vcat(outer_perm_a, inner_perm_a))

        outer_indices_b = Tuple(indices)[mask_b]
        chunks_b = reduce(zip(outer_perm_b, outer_indices_b); init = Dagger.chunks(op.b)) do acc, (d, i)
            selectdim(acc, d, i:i)
        end
        chunks_b = permutedims(chunks_b, vcat(inner_perm_b, outer_perm_b))

        chunks[indices...] = Dagger.treereduce(Dagger.AddComputeOp, map(chunks_a, chunks_b) do chunk_a, chunk_b
            Dagger.@spawn contractor(chunk_a, chunk_b)
        end)
    end

    Dagger.DArray(T, domain, subdomains, chunks, partitioning)
end

function Tenet.contract(
    a::Tensor{Ta,Na,Aa},
    b::Tensor{Tb,Nb,Ab};
    dims = (∩(inds(a), inds(b))),
) where {Ta,Tb,Na,Nb,Aa<:Dagger.DArray{Ta,Na},Ab<:Dagger.DArray{Tb,Nb}}
    ia = inds(a) |> collect
    ib = inds(b) |> collect
    i = ∩(dims, ia, ib)

    ic = setdiff(ia ∪ ib, i isa Base.AbstractVecOrTuple ? i : (i,))::Vector{Symbol}

    data = Dagger._to_darray(Contract(parent(a), ia, parent(b), ib, ic))

    return Tensor(data, ic)
end

end

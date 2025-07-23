module TenetITensorMPSExt

using Tenet
using ITensors: ITensors, ITensor, dim, siteinds, prime
using ITensorMPS: ITensorMPS, linkinds, firstsiteinds

# TODO update code
#= 
# Convert an AbstractMPS to an ITensor MPS
function Base.convert(::Type{ITensorMPS.MPS}, mps::Tenet.AbstractMPS)
    @assert form(mps) isa MixedCanonical "Currently only MixedCanonical MPS conversion is supported"

    ortho_center = form(mps).orthog_center

    itensors = ITensor[]
    for lane in lanes(mps)
        t = Tenet.permutedims(
            tensors(mps; at=lane),
            Vector{Symbol}(
                filter!(
                    !isnothing,
                    [inds(mps; at=Site(lane)), inds(mps; at=lane, dir=:left), inds(mps; at=lane, dir=:right)],
                ),
            ),
        )

        i = Tenet.id(lane)
        site_index = Index(size(mps, inds(mps; at=Site(lane))), "Site,n=$i")
        if i == 1
            link_size = size(mps, inds(mps; at=lane"1", dir=:right))
            link_indices = [Index(link_size, "Link,l=1")]
        else
            # Take index from previous tensor as the left link index
            prev_ind = ITensors.inds(itensors[end])[end]

            if i < length(tensors(mps))
                next_link_size = size(mps, inds(mps; at=lane, dir=:right))
                next_ind = Index(next_link_size, "Link,l=$(i)")
                link_indices = [prev_ind, next_ind]
            else
                link_indices = [prev_ind]
            end
        end
        all_indices = (site_index, link_indices...)

        it = ITensor(parent(t), all_indices...)
        push!(itensors, it)
    end

    itensors_mps = ITensorMPS.MPS(itensors)

    # Set llim and rlim based on the orthogonality center
    if ortho_center isa Lane
        n = Tenet.id(ortho_center)
        itensors_mps.llim = n - 1
        itensors_mps.rlim = n + 1

    elseif ortho_center isa Vector{<:Lane}
        ids = Tenet.id.(ortho_center)

        # For multiple orthogonality centers, set llim and rlim accordingly
        itensors_mps.llim = minimum(ids) - 1
        itensors_mps.rlim = maximum(ids) + 1
    end

    return itensors_mps
end
=#

function Base.convert(::Type{MPS}, itensors_mps::ITensorMPS.MPS)
    llim = itensors_mps.llim
    rlim = itensors_mps.rlim

    # Extract site and link indices
    sites = siteinds(itensors_mps)
    links = linkinds(itensors_mps)

    # Extract the bulk tensors
    arrays_vec = AbstractArray[]
    push!(arrays_vec, ITensors.array(itensors_mps[1], links[1], sites[1]))
    for j in 2:(length(itensors_mps) - 1)
        # Indices are ordered as (left link, right link, site index)
        push!(arrays_vec, ITensors.array(itensors_mps[j], links[j - 1], links[j], sites[j]))
    end
    push!(arrays_vec, ITensors.array(itensors_mps[end], links[end], sites[end]))

    mps = Tenet.MPS(arrays_vec; order=(:l, :r, :o))

    # Map llim and rlim to your MPS's orthogonality center(s)
    mps_form = if llim + 1 == rlim - 1
        Tenet.MixedCanonical(site"$(llim + 1)")
    elseif llim + 1 < rlim - 1
        Tenet.MixedCanonical([site"$j" for j in (llim + 1):(rlim - 1)])
    else
        Tenet.NonCanonical()
    end

    Tenet.unsafe_setform!(mps, mps_form)

    return mps
end

function Base.convert(::Type{MPO}, itensors_mps::ITensorMPS.MPO)
    llim = itensors_mps.llim
    rlim = itensors_mps.rlim

    # Extract site and link indices
    sites = firstsiteinds(itensors_mps)
    links = linkinds(itensors_mps)

    # Extract the bulk tensors
    arrays_vec = AbstractArray[]
    push!(arrays_vec, ITensors.array(itensors_mps[1], links[1], prime(sites[1]), sites[1]))
    for j in 2:(length(itensors_mps) - 1)
        # Indices are ordered as (site index, left link, right link)
        push!(arrays_vec, ITensors.array(itensors_mps[j], links[j - 1], links[j], prime(sites[j]), sites[j]))
    end
    push!(arrays_vec, ITensors.array(itensors_mps[end], links[end], prime(sites[end]), sites[end]))

    mpo = Tenet.MPO(arrays_vec; order=(:l, :r, :o, :i))

    # Map llim and rlim to your MPS's orthogonality center(s)
    mpo_form = if llim + 1 == rlim - 1
        Tenet.MixedCanonical(site"$(llim + 1)")
    elseif llim + 1 < rlim - 1
        Tenet.MixedCanonical([site"$j" for j in (llim + 1):(rlim - 1)])
    else
        Tenet.NonCanonical()
    end

    Tenet.unsafe_setform!(mpo, mpo_form)

    return mpo
end

end

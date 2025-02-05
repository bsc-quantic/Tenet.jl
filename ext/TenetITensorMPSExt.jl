module TenetITensorMPSExt

using Tenet
using Tenet: Tenet, MPS, tensors, form, inds, lanes, Site, Lane
using ITensors: ITensors, ITensor, Index, dim, siteinds
using ITensorMPS: ITensorMPS, linkinds

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

# Convert an ITensor MPS to an MPS
function Base.convert(::Type{MPS}, itensors_mps::ITensorMPS.MPS)
    llim = itensors_mps.llim
    rlim = itensors_mps.rlim

    # Extract site and link indices
    sites = siteinds(itensors_mps)
    links = linkinds(itensors_mps)

    tensors_vec = []
    first_ten = ITensors.array(itensors_mps[1], sites[1], links[1])
    push!(tensors_vec, first_ten)

    # Extract the bulk tensors
    for j in 2:(length(itensors_mps) - 1)
        ten = ITensors.array(itensors_mps[j], sites[j], links[j - 1], links[j]) # Indices are ordered as (site index, left link, right link)
        push!(tensors_vec, ten)
    end
    last_ten = ITensors.array(itensors_mps[end], sites[end], links[end])
    push!(tensors_vec, last_ten)

    mps = Tenet.MPS(tensors_vec)

    # Map llim and rlim to your MPS's orthogonality center(s)
    mps_form = if llim + 1 == rlim - 1
        Tenet.MixedCanonical(Lane(llim + 1))
    elseif llim + 1 < rlim - 1
        Tenet.MixedCanonical([Lane(j) for j in (llim + 1):(rlim - 1)])
    else
        Tenet.NonCanonical()
    end

    mps.form = mps_form

    return mps
end

end

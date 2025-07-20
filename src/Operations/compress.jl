using Muscle

function compress! end

function generic_mps_compress!(tn; maxdim=nothing, threshold=nothing, kwargs...)
    n = count(s -> s isa CartesianSite, Tangles.all_sites_iter(tn))
    for i in 1:(n - 1)
        generic_mps_compress!(tn, bond"$i-$(i+1)"; kwargs...)
    end
    return tn
end

function generic_mps_compress!(tn, _bond; maxdim=nothing, threshold=nothing, kwargs...)
    @argcheck !isnothing(maxdim) || !isnothing(threshold) "Either `maxdim` or `threshold` must be specified."
    @argcheck isnothing(maxdim) || maxdim > 0 "maxdim must be a positive integer."
    @argcheck isnothing(threshold) || threshold > 0 "Threshold must be positive."

    bondind = ind_at(tn, _bond)
    canonize!(tn, BondCanonical(_bond))

    old_u = tensor_at(tn, site"$i")
    old_v = tensor_at(tn, site"$(i + 1)")
    old_s = tensor_at(tn, LambdaSite(_bond))

    new_u = old_u
    new_v = old_v
    new_s = old_s

    # use `maxdim` to truncate the singular values
    if !isnothing(maxdim)
        new_u = view(new_u, bonind => 1:min(maxdim, length(old_s)))
        new_v = view(new_v, bonind => 1:min(maxdim, length(old_s)))
        new_s = view(new_s, bonind => 1:min(maxdim, length(old_s)))
    end

    # use `threshold` to truncate the singular values
    if !isnothing(threshold)
        keep = findall(x -> abs(x) > threshold, S)
        new_u = view(new_u, bondind => keep)
        new_v = view(new_v, bondind => keep)
        new_s = view(new_s, bondind => keep)
    end

    # update the tensor network
    @unsafe_region tn begin
        replace_tensor!(tn, old_u, new_u)
        replace_tensor!(tn, old_v, new_v)
        replace_tensor!(tn, old_s, new_s)
    end

    return tn
end

compress!(tn::MPS; kwargs...) = generic_mps_compress!(tn, kwargs...)
compress!(tn::MPO; kwargs...) = generic_mps_compress!(tn, kwargs...)

using Muscle

"""
    compress!(tn, args...; kwargs...)

Truncate the bonds of the Tensor Network with minimal error.

## Keyword arguments

  - `maxdim`: maximum bond dimension to keep.
  - `threshold`: threshold for truncating singular values.
"""
function compress! end

compress(tn, args...; kwargs...) = compress!(copy(tn), args...; kwargs...)

# TODO should we rename `compress!` to `truncate!`?
const truncate! = compress!
const truncate = compress

function generic_mps_compress!(tn; kwargs...)
    n = count(s -> s isa CartesianSite, Tangles.all_sites_iter(tn))
    for i in 1:(n - 1)
        generic_mps_compress!(tn, bond"$i-$(i+1)"; kwargs...)
    end
    return tn
end

generic_mps_compress!(tn, _bond; kwargs...) = generic_mps_compress!(form(tn), tn, _bond; kwargs...)

function generic_mps_compress!(::CanonicalForm, tn, _bond; maxdim=nothing, threshold=nothing, kwargs...)
    @argcheck !isnothing(maxdim) || !isnothing(threshold) "Either `maxdim` or `threshold` must be specified"
    @argcheck isnothing(maxdim) || maxdim > 0 "maxdim must be a positive integer"
    @argcheck isnothing(threshold) || threshold > 0 "Threshold must be positive"

    site_u, site_v = minmax(sites(_bond)...)
    bondind = ind_at(tn, _bond)
    canonize!(tn, BondCanonical(_bond))

    old_u = tensor_at(tn, site_u)
    old_v = tensor_at(tn, site_v)
    old_s = tensor_at(tn, LambdaSite(_bond))

    new_u = old_u
    new_v = old_v
    new_s = old_s

    # use `maxdim` to truncate the singular values
    if !isnothing(maxdim)
        new_u = view(new_u, bondind => 1:min(maxdim, length(old_s)))
        new_v = view(new_v, bondind => 1:min(maxdim, length(old_s)))
        new_s = view(new_s, bondind => 1:min(maxdim, length(old_s)))
    end

    # use `threshold` to truncate the singular values
    if !isnothing(threshold)
        keep = findall(x -> abs(x) > threshold, new_s)
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

function generic_mps_compress!(::VidalGauge, tn, _bond; maxdim=nothing, threshold=nothing, kwargs...)
    @argcheck !isnothing(maxdim) || !isnothing(threshold) "Either `maxdim` or `threshold` must be specified"
    @argcheck isnothing(maxdim) || maxdim > 0 "maxdim must be a positive integer"
    @argcheck isnothing(threshold) || threshold > 0 "Threshold must be positive"

    sitel, siter = minmax(sites(_bond)...)
    Λ = tensor_at(tn, LambdaSite(_bond))

    if !isnothing(maxdim)
        keep = 1:min(maxdim, length(Λ))
        Λ = @view Λ[only(inds(Λ)) => keep]
    end

    if !isnothing(threshold)
        keep = findall(x -> abs(x) > threshold, Λ)
    end

    Tangles.slice!(tn, ind_at(tn, _bond), keep)

    return tn
end

compress!(tn::MPS, args...; kwargs...) = generic_mps_compress!(tn, args...; kwargs...)
compress!(tn::MPO, args...; kwargs...) = generic_mps_compress!(tn, args...; kwargs...)

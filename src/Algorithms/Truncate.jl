"""
    truncate(tn, bond; threshold = nothing, maxdim = nothing)

Like [`truncate!`](@ref), but returns a new Tensor Network instead of modifying the original one.
"""
Base.truncate(tn::AbstractTensorNetwork, args...; kwargs...) = truncate!(copy(tn), args...; kwargs...)

"""
    truncate!(tn, bond; threshold = nothing, maxdim = nothing)

Truncate the dimension of the virtual `bond`` of an Tensor Network. Dispatches to the appropriate method based on the [`form`](@ref) of the Tensor Network:

  - If the Tensor Network is in the [`MixedCanonical`](@ref) form, the bond is truncated by moving the orthogonality center to the bond and keeping the `maxdim` largest **Schmidt coefficients** or those larger than `threshold`.
  - If the Tensor Network is in the [`Canonical`](@ref) form, the bond is truncated by keeping the `maxdim` largest **Schmidt coefficients** or those larger than `threshold`, and then recanonizing the Tensor Network.
  - If the Tensor Network is in the [`NonCanonical`](@ref) form, the bond is truncated by contracting the bond, performing an SVD and keeping the `maxdim` largest **singular values** or those larger than `threshold`.

# Notes

  - Either `threshold` or `maxdim` must be provided. If both are provided, `maxdim` is used.
"""
function truncate!(tn, bond; threshold=nothing, maxdim=nothing, kwargs...)
    all(isnothing, (threshold, maxdim)) && return tn
    return truncate!(form(tn), tn, bond; threshold, maxdim, kwargs...)
end

"""
    truncate!(::NonCanonical, tn, bond; threshold, maxdim, compute_local_svd=true)

Truncate the dimension of the virtual `bond` of a [`NonCanonical`](@ref) Tensor Network by contracting the bond, performing an SVD and keeping the `maxdim` largest **singular values** or those larger than `threshold`.

# Keyword Arguments

  - `threshold`: The threshold to truncate the bond dimension.
  - `maxdim`: The maximum bond dimension to keep.
  - `compute_local_svd`: Whether to compute the local SVD of the bond. If `true`, it will contract the bond and perform a SVD to get the local singular values. Defaults to `true`.
"""
function truncate!(::NonCanonical, tn, bond; threshold, maxdim, compute_local_svd=true)
    virtualind = inds(tn; bond)
    tₗ = tensors(tn; at=min(bond...))
    tᵣ = tensors(tn; at=max(bond...))

    u, s, v = if compute_local_svd
        t = contract(tₗ, tᵣ; dims=[virtualind])

        left_inds = filter(!=(virtualind), inds(tₗ))
        right_inds = filter(!=(virtualind), inds(tᵣ))
        svd(t; left_inds, right_inds, virtualind)
    else
        s = tensors(tn; bond)
        tₗ, s, tᵣ
    end

    maxdim = isnothing(maxdim) ? size(tn, virtualind) : min(maxdim, length(s))

    extent = if isnothing(threshold)
        1:maxdim
    else
        # Find the first index where the condition is met
        found_index = findfirst(1:maxdim) do i
            abs(s[i]) < threshold
        end

        # If no index is found, return 1:length(s), otherwise calculate the range
        1:(isnothing(found_index) ? maxdim : found_index - 1)
    end

    slice!(tn, virtualind, extent)

    return tn
end

function truncate!(::MixedCanonical, tn, bond; kwargs...)
    # move orthogonality center to bond
    # TODO implement BondCanonical form so that we have no need to compute the local SVD
    canonize!(tn, MixedCanonical(collect(bond)))
    return truncate!(NonCanonical(), tn, bond; compute_local_svd=true, kwargs...)
end

# TODO propagate loss of canonicalization?
"""
    truncate!(::Canonical, tn, bond; kwargs...)

Truncate the dimension of the virtual `bond` of a [`Canonical`](@ref) Tensor Network by keeping the `maxdim` largest
**Schmidt coefficients** or those larger than `threshold`, and then canonizes the Tensor Network if `canonize` is `true`.
"""
function truncate!(::Canonical, tn, bond; kwargs...)
    truncate!(NonCanonical(), tn, bond; compute_local_svd=false, kwargs...)
    return tn
end

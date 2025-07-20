using ArgCheck
using Muscle

"""
    canonize!(tn, form)

Change the canonical form specified by `form`.

See also: [`form`](@ref).
"""
function canonize! end

canonize(tn, args...; kwargs...) = canonize!(copy(tn), args...; kwargs...)

# shortcuts
canonize!(tn, orthog_center::Site; kwargs...) = canonize!(tn, MixedCanonical(orthog_center); kwargs...)
canonize!(tn, orthog_center::Vector{<:Site}; kwargs...) = canonize!(tn, MixedCanonical(orthog_center); kwargs...)
canonize!(tn, orthog_center::Bond; kwargs...) = canonize!(tn, BondCanonical(orthog_center); kwargs...)
canonize!(tn, ::NonCanonical; kwargs...) = unsafe_setform!(tn, NonCanonical())
canonize!(tn, old_form, ::NonCanonical; kwargs...) = unsafe_setform!(tn, NonCanonical())

# auxiliar functions
function generic_canonize_site!(tn, _site::Site, _bond::Bond; method=:qr)
    @assert hassite(_bond, _site)
    @assert hassite(tn, _site)
    @assert hasbond(tn, _bond)

    # A it the tensor where we perform the factorization, but B is also affected by the gauge transformation
    A = tensor(tn; at=_site)
    B = tensor(tn; at=only(filter(!=(_site), sites(_bond))))

    ind_iso_dir = ind(tn; at=_bond)
    inds_a_only = filter(!=(ind_iso_dir), inds(A))
    inds_b_only = Index[ind_iso_dir]
    ind_virtual = Index(gensym(:tmp))

    # cuTensorNet runs into an error when the size of the virtual index is 1
    size(tn, ind_iso_dir) == 1 && return tn

    if method === :svd
        U, s, V = tensor_svd_thin(A; inds_u=inds_a_only, inds_v=inds_b_only, ind_s=ind_virtual)

        # absorb singular values
        V = binary_einsum(s, V; dims=Index[])

        # contract V against next lane tensor
        V = binary_einsum(B, V)

        # rename back bond index
        U = replace(U, ind_virtual => ind_iso_dir)
        V = replace(V, ind_virtual => ind_iso_dir)

        # replace old tensors with new gauged ones
        replace!(tn, A => U)
        replace!(tn, B => V)

    elseif method === :qr
        Q, R = tensor_qr_thin(A; inds_q=inds_a_only, inds_r=inds_b_only, ind_virtual=ind_virtual)

        # contract R against next lane tensor
        R = binary_einsum(R, B)

        # rename back bond index
        Q = replace(Q, ind_virtual => ind_iso_dir)
        R = replace(R, ind_virtual => ind_iso_dir)

        # replace old tensors with new gauged ones
        replace!(tn, A => Q)
        replace!(tn, B => R)

    else
        throw(ArgumentError("Unknown factorization method=:$method"))
    end

    return tn
end

function generic_bond_canonize_site!(tn, _site::Site, _bond::Bond)
    @assert hassite(_bond, _site)
    @assert hassite(tn, _site)
    @assert hasbond(tn, _bond)
    @assert !hassite(tn, LambdaSite(_bond))

    # A it the tensor where we perform the factorization, but B is also affected by the gauge transformation
    A = tensor(tn; at=_site)
    B = tensor(tn; at=only(filter(!=(_site), sites(_bond))))

    ind_iso_dir = ind(tn; at=_bond)
    inds_a_only = filter(!=(ind_iso_dir), inds(A))
    inds_b_only = Index[ind_iso_dir]
    ind_virtual = Index(gensym(:tmp))

    # cuTensorNet runs into an error when the size of the virtual index is 1
    size(tn, ind_iso_dir) == 1 && return tn

    # it's just like `generic_canonize_site!(; method=:svd)` but we do not absorb the singular values
    U, s, V = tensor_svd_thin(A; inds_u=inds_a_only, inds_v=inds_b_only, ind_s=ind_virtual)

    # contract V against next lane tensor
    V = binary_einsum(B, V)

    # rename back bond index
    U = replace(U, ind_virtual => ind_iso_dir)
    V = replace(V, ind_virtual => ind_iso_dir)
    s = replace(s, ind_virtual => ind_iso_dir)

    # replace old tensors with new gauged ones
    replace!(tn, A => U)
    replace!(tn, B => V)

    # add the singular values as a new tensor
    addtensor!(tn, s)
    setsite!(tn, s, LambdaSite(_bond))

    return tn
end

canonize!(tn::AbstractMPS, i::Integer; kwargs...) = canonize!(tn, MixedCanonical(site"$i"))

## `MatrixProductState` / `MatrixProductOperator`
generic_mps_canonize!(tn, new_form) = generic_mps_canonize!(tn, CanonicalForm(tn), new_form)

function generic_mps_canonize!(tn, ::NonCanonical, new_form::MixedCanonical)
    unsafe_setform!(tn, MixedCanonical(sites(tn)))
    generic_mps_canonize!(tn, new_form)
end

function generic_mps_canonize!(tn, old_form::MixedCanonical, new_form::MixedCanonical)
    old_form == new_form && return tn

    # TODO maybe use sth different to `.id`?
    src_left, src_right = site(min_orthog_center(old_form)).id[1], site(max_orthog_center(old_form)).id[1]
    dst_left, dst_right = site(min_orthog_center(new_form)).id[1] - 1, site(max_orthog_center(new_form)).id[1] + 1

    # left-to-right QR sweep (left-canonical tensors)
    for i in src_left:dst_left
        bond = bond"$i - $(i + 1)"
        generic_canonize_site!(tn, site"$i", bond; method=:qr)
    end

    # right-to-left QR sweep (right-canonical tensors)
    for i in src_right:-1:dst_right
        bond = bond"$(i - 1) - $i"
        generic_canonize_site!(tn, site"$i", bond; method=:qr)
    end

    unsafe_setform!(tn, copy(new_form))
    return tn
end

function generic_mps_canonize!(tn, old_form::MixedCanonical, new_form::BondCanonical)
    old_a, old_b = min_orthog_center(old_form), max_orthog_center(old_form)
    new_a, new_b = sites(orthog_center(new_form))

    old_min, old_max = minmax(old_a, old_b)
    new_min, new_max = minmax(new_a, new_b)

    if old_max <= new_min
        # canonize to the right
        canonize!(tn, MixedCanonical(new_min))
        generic_bond_canonize_site!(tn, new_min, orthog_center(new_form))

    elseif new_max <= old_min
        # canonize to the left
        canonize!(tn, MixedCanonical(new_max))
        generic_bond_canonize_site!(tn, new_max, orthog_center(new_form))

    elseif old_min <= new_min && new_max <= old_max
        # canonize to one of the sites and then bond-canonize
        canonize!(tn, MixedCanonical(new_min))
        generic_bond_canonize_site!(tn, new_min, orthog_center(new_form))

    else
        throw(ArgumentError("Cannot canonize from $old_form to $new_form"))
    end

    unsafe_setform!(tn, new_form)
    return tn
end

function generic_mps_canonize!(tn, old_form::BondCanonical, new_form::MixedCanonical)
    old_a, old_b = sites(orthog_center(old_form))
    new_a, new_b = min_orthog_center(new_form), max_orthog_center(new_form)

    old_min, old_max = minmax(old_a, old_b)
    new_min, new_max = minmax(new_a, new_b)

    _bond = orthog_center(old_form)
    s = tensor_at(tn, LambdaSite(_bond))
    rmtensor!(tn, s)

    if old_max <= new_min
        # absorb to the right to form `MixedCanonical(old_max)` and then mixed canonize
        _tensor = tensor_at(tn, old_max)

        # TODO `hadamard!` gives problems when saving `tn` before the current canonical form (i.e. `canonize(tn, ...)`)
        # Muscle.hadamard!(_tensor, _tensor, s)
        replace_tensor!(tn, _tensor, hadamard(_tensor, s))

        unsafe_setform!(tn, MixedCanonical(old_max))
        canonize!(tn, new_form)

    elseif new_max <= old_min
        # absorb to the left to form `MixedCanonical(old_min)` and then mixed canonize
        _tensor = tensor_at(tn, old_min)

        # TODO `hadamard!` gives problems when saving `tn` before the current canonical form (i.e. `canonize(tn, ...)`)
        # Muscle.hadamard!(_tensor, _tensor, s)
        replace_tensor!(tn, _tensor, hadamard(_tensor, s))

        unsafe_setform!(tn, MixedCanonical(old_min))
        canonize!(tn, new_form)
    else
        throw(ArgumentError("Cannot canonize from $old_form to $new_form"))
    end

    unsafe_setform!(tn, new_form)
    return tn
end

function generic_mps_canonize!(tn, old_form::BondCanonical, new_form::BondCanonical)
    old_form == new_form && return tn

    old_a, old_b = sites(orthog_center(old_form))
    new_a, new_b = sites(orthog_center(new_form))

    old_min, old_max = minmax(old_a, old_b)
    new_min, new_max = minmax(new_a, new_b)

    if old_max <= new_min
        # canonize to the right
        canonize!(tn, MixedCanonical(new_min))
        generic_bond_canonize_site!(tn, new_min, orthog_center(new_form))

    elseif new_max <= old_min
        # canonize to the left
        canonize!(tn, MixedCanonical(new_max))
        generic_bond_canonize_site!(tn, new_max, orthog_center(new_form))

    else
        throw(ArgumentError("Cannot canonize from $old_form to $new_form"))
    end

    unsafe_setform!(tn, new_form)
    return tn
end

function generic_mps_canonize!(tn, ::NonCanonical, new_form::BondCanonical)
    unsafe_setform!(tn, MixedCanonical(sites(tn)))
    canonize!(tn, new_form)
    return tn
end

canonize!(tn::MPS, new_form::CanonicalForm; kwargs...) = generic_mps_canonize!(tn, new_form; kwargs...)
canonize!(tn::MPO, new_form::CanonicalForm; kwargs...) = generic_mps_canonize!(tn, new_form; kwargs...)

# TODO
function canonize!(tn::AbstractMPO, old_form::VidalGauge, new_form::MixedCanonical; kwargs...)
    for i in 1:(min_orthog_center(new_form) - 1)
        bond = bond"$i - $(i + 1)"
        # TODO absorb!(tn, bond, :right)
    end

    for i in nsites(tn):-1:(max_orthog_center(new_form) + 1)
        bond = bond"$(i - 1) - $i"
        # TODO absorb!(tn, bond, :left)
    end

    # a sweep is need to fully propagate the effects of truncation
    # TODO probably there is a better way to propagate these effects
    # sweep && canonize!(NonCanonical(), tn, targetform)

    unsafe_setform!(tn, copy(targetform))
    return tn
end

# TODO
function canonize!(tn::AbstractMPO, old_form::VidalGauge, new_form::VidalGauge; kwargs...) end

# TODO
function canonize!(tn::AbstractMPO, old_form::NonCanonical, new_form::VidalGauge; kwargs...)
    # right-to-left QR sweep, get right-canonical tensors
    canonize!(tn, MixedCanonical(site"1"))

    # left-to-right SVD sweep, get left-canonical tensors and singular values without reversing
    for i in 1:(nsites(tn) - 1)
        bond = bond"$i - $(i + 1)"
        generic_canonize_site!(tn, site"$i", bond; method=:svd)

        # extract the singular values and contract them with the next tensor
        # NOTE do not remove them, since they will be needed but TN can in be in a inconsistent state while processing
        Λᵢ = tensor(tn; at=bond)

        Aᵢ₊₁ = tensor(tn; at=site"$(i + 1)")
        replace!(tn, Aᵢ₊₁ => contract(Aᵢ₊₁, Λᵢ; dims=Index[]))
    end

    # tensors at i in "A" form, need to contract (Λᵢ)⁻¹ with A to get Γᵢ
    for i in 2:nsites(tn)
        bond = bond"$(i - 1) - $i"
        Λᵢ = tensor(tn; at=bond)
        Aᵢ = tensor(tn; at=site"$i")
        Λᵢ⁻¹ = Tensor(diag(pinv(Diagonal(parent(Λᵢ)); atol=1e-64)), inds(Λᵢ))
        Γᵢ = contract(Aᵢ, Λᵢ⁻¹; dims=Index[])
        replace!(tn, Aᵢ => Γᵢ)
    end

    unsafe_setform!(tn, Canonical())
    return tn
end

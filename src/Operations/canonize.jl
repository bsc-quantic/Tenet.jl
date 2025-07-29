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

## `MatrixProductState` / `MatrixProductOperator`
canonize!(tn::AbstractMPS, i::Integer; kwargs...) = canonize!(tn, MixedCanonical(site"$i"))
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

function generic_mps_canonize!(tn, ::NonCanonical, ::VidalGauge)
    # first mixed-canonize to the first site (or any other site really)
    canonize!(tn, MixedCanonical(site"1"))

    # and retrigger Vidal canonization
    canonize!(tn, VidalGauge())
end

function generic_mps_canonize!(tn, old_form::BondCanonical, ::VidalGauge)
    # absorb lambda to one of the sites
    canonize!(tn, MixedCanonical(min_orthog_center(tn)))

    # retrigger canonization
    canonize!(tn, VidalGauge())
    return tn
end

function generic_mps_canonize!(tn, old_form::MixedCanonical, ::VidalGauge)
    if min_orthog_center(old_form) != max_orthog_center(old_form)
        canonize!(tn, MixedCanonical(min_orthog_center(old_form)))
        old_form = CanonicalForm(tn)
    end

    # orthogonality center is a single site, so we propagate the lambdas from there
    oc = only(Tuple(orthog_center(old_form)))
    ncartsites = count(s -> s isa CartesianSite, all_sites(tn))

    # right-to-left SVD sweep, get right-canonical tensors and singular values without reversing
    for i in oc:-1:2
        # bond-canonize locally
        generic_bond_canonize_site!(tn, site"$i", bond"$(i-1)-$i")

        # extract the singular values and contract them with the next tensor
        # NOTE do not remove them, since they will be needed but TN can in be in a inconsistent state while processing
        Λ = tensor_at(tn, lambda"$(i-1)-$i")
        A = tensor_at(tn, site"$i - 1")

        Anew = hadamard(A, Λ)
        replace_tensor!(tn, A, Anew)
    end

    # left-to-right SVD sweep, get left-canonical tensors and singular values without reversing
    for i in oc:(ncartsites - 1)
        # bond-canonize locally
        generic_bond_canonize_site!(tn, site"$i", bond"$i-$(i+1)")

        # extract the singular values and contract them with the next tensor
        # NOTE do not remove them, since they will be needed but TN can in be in a inconsistent state while processing
        Λ = tensor_at(tn, lambda"$i-$(i+1)")
        A = tensor_at(tn, site"$i + 1")
        Anew = hadamard(A, Λ)
        replace_tensor!(tn, A, Anew)
    end

    # tensors are in "A" form, need to contract (Λᵢ)⁻¹ with A to get Γᵢ
    for lambda_site in Iterators.filter(s -> s isa LambdaSite, all_sites(tn))
        left_site, right_site = minmax(sites(bond(lambda_site))...)
        Λ = tensor_at(tn, lambda_site)
        Al = tensor_at(tn, left_site)
        Ar = tensor_at(tn, right_site)

        Λ⁻¹ = Tensor(diag(pinv(Diagonal(parent(Λ)); atol=1e-64)), inds(Λ))

        Γl = hadamard(Al, Λ⁻¹)
        replace_tensor!(tn, Al, Γl)

        Γr = hadamard(Ar, Λ⁻¹)
        replace_tensor!(tn, Ar, Γr)
    end

    unsafe_setform!(tn, VidalGauge())
    return tn
end

function generic_mps_canonize!(tn, ::VidalGauge, ::VidalGauge)
    @warn "Ignoring canonization from `VidalGauge` to `VidalGauge` form for $(typeof(tn)). If you want to force \
    canonization, please call `canonize!(tn, NonCanonical())` first and then run again this function."
    return tn
end

function generic_mps_canonize!(tn, ::VidalGauge, new_form::BondCanonical)
    left_boundary, right_boundary = minmax(sites(orthog_center(new_form))...)

    for lambda_site in Iterators.filter(s -> s isa LambdaSite, all_sites(tn))
        left_lambda_site, right_lambda_site = minmax(sites(bond(lambda_site))...)
        if left_lambda_site == left_boundary && right_lambda_site == right_boundary
            # do not absorb orthogonality center
            continue

        elseif right_lambda_site <= left_boundary
            # absorb lambda do the right to form the left canonical tensor
            Λ = tensor_at(tn, lambda_site)
            Γ = tensor_at(tn, right_lambda_site)
            Γnew = Muscle.hadamard(Γ, Λ)
            replace_tensor!(tn, Γ, Γnew)

        elseif right_boundary <= left_lambda_site
            # absorb lambda do the left to form the right canonical tensor
            Λ = tensor_at(tn, lambda_site)
            Γ = tensor_at(tn, left_lambda_site)
            Γnew = Muscle.hadamard(Γ, Λ)
            replace_tensor!(tn, Γ, Γnew)

        else
            error("Lambda sites are in a incoherent state")
        end
    end

    unsafe_setform!(tn, new_form)
    return tn
end

function generic_mps_canonize!(tn, ::VidalGauge, new_form::MixedCanonical)
    left_boundary, right_boundary = min_orthog_center(new_form), max_orthog_center(new_form)

    for lambda_site in Iterators.filter(s -> s isa LambdaSite, all_sites(tn))
        left_lambda_site, right_lambda_site = minmax(sites(lambda_site)...)

        if left_boundary <= left_lambda_site < right_lambda_site <= right_boundary
            # absorb lambda equally
            Λ = tensor_at(tn, lambda_site)
            Λsqrt = sqrt.(Λ)

            Γl = tensor_at(tn, left_lambda_site)
            Γlnew = Muscle.hadamard(Γl, Λsqrt)
            replace_tensor!(tn, Γlnew, Γl)

            Γr = tensor_at(tn, right_lambda_site)
            Γrnew = Muscle.hadamard(Γr, Λsqrt)
            replace_tensor!(tn, Γrnew, Γr)

        elseif right_lambda_site <= left_boundary
            # absorb lambda to the right to form the left canonical tensor
            Λ = tensor_at(tn, lambda_site)
            Γ = tensor_at(tn, right_lambda_site)
            Γnew = Muscle.hadamard(Γ, Λ)
            replace_tensor!(tn, Γnew, Γ)

        elseif right_boundary <= left_lambda_site
            # absorb lambda to the left to form the right canonical tensor
            Λ = tensor_at(tn, lambda_site)
            Γ = tensor_at(tn, left_lambda_site)
            Γnew = Muscle.hadamard(Γ, Λ)
            replace_tensor!(tn, Γnew, Γ)

        else
            error("Lambda sites are in a incoherent state")
        end
    end

    unsafe_setform!(tn, new_form)
    return tn
end

function generic_mps_canonize!(tn, ::VidalGauge, ::NonCanonical)
    for lambda_site in Iterators.filter(s -> s isa LambdaSite, all_sites(tn))
        left_lambda_site, right_lambda_site = minmax(sites(orthog_center(new_form))...)

        # absorb lambda equally
        Λ = tensor_at(tn, lambda_site)
        Λsqrt = sqrt.(Λ)

        Γl = tensor_at(tn, left_lambda_site)
        Γlnew = Muscle.hadamard(Γl, Λsqrt)
        replace_tensor!(tn, Γlnew, Γl)

        Γr = tensor_at(tn, right_lambda_site)
        Γrnew = Muscle.hadamard(Γr, Λsqrt)
        replace_tensor!(tn, Γrnew, Γr)
    end

    unsafe_setform!(tn, new_form)
    return tn
end

canonize!(tn::MPS, new_form::CanonicalForm; kwargs...) = generic_mps_canonize!(tn, new_form; kwargs...)
canonize!(tn::MPO, new_form::CanonicalForm; kwargs...) = generic_mps_canonize!(tn, new_form; kwargs...)

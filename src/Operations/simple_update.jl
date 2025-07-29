using ArgCheck
using Muscle
using QuantumTags

function simple_update! end

# auxiliar functions
function acting_sites(operator::Tensor)
    @argcheck all(isplug, inds(operator)) "Operator indices must be plugs to be treated as an operator"

    target_plugs = plugs(operator)
    target_plugs_dual = filter(isdual, target_plugs)
    target_plugs_normal = filter(!isdual, target_plugs)

    @argcheck issetequal(target_plugs_normal, adjoint.(target_plugs_dual)) "Operator must have same input and output plugs"

    return unique(site.(target_plugs_dual))
end

function generic_simple_update_1site!(tn, operator)
    op_sites = acting_sites(operator)
    @assert 1 == length(op_sites) <= 2 "Operator must act on one site"
    op_site = only(op_sites)
    @argcheck hasplut(tn, plug"$op_site") "Operator plug must be present"

    _tensor = tensor_at(tn, op_site)
    tmp_contracting_ind = Index(gensym(:tmp))
    tensor = replace(_tensor, ind_at(tn, plug"$op_site") => tmp_contracting_ind)
    operator = replace(operator, Index(plug"$op_site'") => tmp_contracting_ind)
    new_tensor = binary_einsum(tensor, operator)
    replace_tensor!(tn, _tensor, new_tensor)

    return tn
end

function generic_simple_update!(tn, operator; maxdim=nothing)
    op_sites = acting_sites(operator)
    @assert length(op_sites) == 2 "Operator must act on two sites"
    @argcheck all(Base.Fix1(hasplug, tn), Plug.(op_sites; isdual=false)) "Operator plugs must be present in the MPS"

    sitel, siter = minmax(op_sites...)
    Al = tensor_at(tn, sitel)
    Ar = tensor_at(tn, siter)

    tmp_contracting_ind_l = Index(gensym(:tmp))
    tmp_contracting_ind_r = Index(gensym(:tmp))

    Al = replace(Al, ind_at(tn, plug"$sitel") => tmp_contracting_ind_l)
    Ar = replace(Ar, ind_at(tn, plug"$siter") => tmp_contracting_ind_r)

    operator = replace(
        operator, Index(plug"$sitel'") => tmp_contracting_ind_l, Index(plug"$siter'") => tmp_contracting_ind_r
    )

    Alnew, Arnew = Muscle.simple_update(
        Al,
        tmp_contracting_ind_l, # ind_physical_left
        Ar,
        tmp_contracting_ind_r, # ind_physical_right
        ind_at(tn, bond"$sitel-$siter"), # ind_bond
        operator,
        Index(plug"$sitel"), # ind_physical_op_left
        Index(plug"$siter"); # ind_physical_op_right
        maxdim,
        absorb=Muscle.AbsorbEqually(),
    )

    # fix the index renaming of `Muscle.simple_update`
    # TODO fix it better in Muscle?
    Alnew = replace(Alnew, tmp_contracting_ind_l => ind_at(tn, plug"$sitel"))
    Arnew = replace(Arnew, tmp_contracting_ind_r => ind_at(tn, plug"$site_b"))

    @unsafe_region tn begin
        replace_tensor!(tn, Al, Alnew)
        replace_tensor!(tn, Ar, Arnew)
    end

    return tn
end

# TODO make `lambda_left`, `lambda_right` vectors so that we can use it for the "BP-Simple Update" procedure
function generic_simple_update_vidal!(tn, operator; maxdim=nothing, lambda_left=nothing, lambda_right=nothing)
    op_sites = acting_sites(operator)
    @assert length(op_sites) == 2 "Operator must act on two sites"
    @argcheck all(Base.Fix1(hasplug, tn), Plug.(op_sites; isdual=false)) "Operator plugs must be present in the MPS"

    sitel, siter = minmax(op_sites...)
    Γl = tensor_at(tn, sitel)
    Γr = tensor_at(tn, siter)

    # absorb inner lambda
    if hassite(tn, lambda"$sitel-$siter")
        Λ = tensor_at(tn, lambda"$sitel-$siter")
        Γl = hadamard(Γl, Λ)
    end

    # absorb exterior lambdas to form a local mixed canonical form
    if !isnothing(lambda_left)
        Λl = tensor_at(tn, lambda_left)
        Γl = hadamard(Γl, Λl)
    end

    if !isnothing(lambda_right)
        Λr = tensor_at(tn, lambda_right)
        Γr = hadamard(Γr, Λr)
    end

    # perform simple update procedure
    tmp_contracting_ind_l = Index(gensym(:tmp))
    tmp_contracting_ind_r = Index(gensym(:tmp))

    Γl = replace(Γl, ind_at(tn, plug"$sitel") => tmp_contracting_ind_l)
    Γr = replace(Γr, ind_at(tn, plug"$siter") => tmp_contracting_ind_r)

    operator = replace(
        operator, Index(plug"$sitel'") => tmp_contracting_ind_l, Index(plug"$siter'") => tmp_contracting_ind_r
    )

    Γlnew, Λ, Γrnew = Muscle.simple_update(
        Γl,
        tmp_contracting_ind_l, # ind_physical_a
        Γr,
        tmp_contracting_ind_r, # ind_physical_b
        ind_at(tn, bond"$sitel-$siter"), # ind_bond_ab
        operator,
        Index(plug"$sitel"), # ind_physical_op_a
        Index(plug"$siter"); # ind_physical_op_b
        maxdim,
        absorb=Muscle.DontAbsorb(),
    )

    # fix the index renaming of `Muscle.simple_update`
    # TODO fix it better in Muscle?
    Γlnew = replace(Γlnew, tmp_contracting_ind_l => ind_at(tn, plug"$sitel"))
    Γrnew = replace(Γrnew, tmp_contracting_ind_r => ind_at(tn, plug"$siter"))

    # absorb (pseudo)inverse lambdas to regenerate gammas
    Λ⁻¹ = Tensor(diag(pinv(Diagonal(parent(Λ)); atol=1e-64)), inds(Λ))
    hadamard!(Γlnew, Γlnew, Λ⁻¹)
    hadamard!(Γrnew, Γrnew, Λ⁻¹)

    # TODO FIX HADAMARD! HERE
    if !isnothing(lambda_left)
        Λl = tensor_at(tn, lambda_left)
        Λl⁻¹ = Tensor(diag(pinv(Diagonal(parent(Λl)); atol=1e-64)), inds(Λl))
        hadamard!(Γlnew, Γlnew, Λl⁻¹)
    end

    # TODO FIX HADAMARD! HERE
    if !isnothing(lambda_right)
        Λr = tensor_at(tn, lambda_right)
        Λr⁻¹ = Tensor(diag(pinv(Diagonal(parent(Λr)); atol=1e-64)), inds(Λr))
        hadamard!(Γrnew, Γrnew, Λr⁻¹)
    end

    # update tensors in the tensor network
    @unsafe_region tn begin
        replace_tensor!(tn, Γl, Γlnew)
        replace_tensor!(tn, Γr, Γrnew)
    end

    if hassite(tn, lambda"$sitel-$siter")
        replace_tensor!(tn, tensor_at(tn, lambda"$sitel-$siter"), Λ)
    else
        addtensor!(tn, Λ)
        setsite!(tn, Λ, lambda"$sitel-$siter")
    end

    return tn
end

simple_update!(tn, operator::Tensor; kwargs...) = generic_simple_update!(tn, operator; kwargs...)

## `MPS`
function simple_update!(tn::MPS, operator::Tensor; kwargs...)
    op_sites = acting_sites(operator)

    if form(tn) isa VidalGauge
        # TODO fix this for "BP - Simple Update" procedure
        op_site_min, op_site_max = minmax(op_sites...)
        op_site_min_idx = only(Tuple(op_site_min))
        op_site_max_idx = only(Tuple(op_site_max))

        ncartsites = count(s -> s isa CartesianSite, all_sites(tn))

        lambda_left = op_site_min != site"1" ? lambda"$(op_site_min_idx - 1) - $op_site_min_idx" : nothing
        lambda_right = op_site_max != site"$ncartsites" ? lambda"$op_site_max_idx - $(op_site_max_idx - 1)" : nothing

        generic_simple_update_vidal!(tn, operator; lambda_left, lambda_right, kwargs...)
    else
        # move orthogonality center to operator sites
        canonize!(tn, MixedCanonical(op_sites))

        # perform the simple update routine
        generic_simple_update!(tn, operator; kwargs...)
    end

    return tn
end

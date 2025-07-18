"""
    entropy_vonneumann(psi)

Calculate the Von Neumann entropy of an MPS `psi`.

See also: [`entropy_vonneumann!`](@ref).
"""
entropy_vonneumann(psi) = entropy_vonneumann!(copy(psi))

"""
    entropy_vonneumann!(psi)

Calculate the Von Neumann entropy of an MPS `psi` by performing a singular value decomposition (SVD) on the tensors of the MPS.

!!! note

    This function is marked as `!` (in-place) because it modifies the gauge of the MPS during the calculation.

!!! warning

    The MPS should be normalized before calling this function. The function does not normalize the MPS internally.

See also: [`entropy_vonneumann`](@ref).
"""
function entropy_vonneumann!(psi::MPS)
    N = nsites(psi)

    entropies = zeros(Float64, nbonds(psi))

    canonize!(psi, MixedCanonical(site"1"))
    for i in 1:(nsites(psi) - 1)
        # A it the tensor where we perform the factorization, but B is also affected by the gauge transformation
        A = psi[i]
        B = psi[i + 1]

        # alternatively, we can just use `psi[bond"$i - $(i+1)"`
        ind_iso_dir = only(intersect(inds(A), inds(B)))
        inds_a_only = filter(!=(ind_iso_dir), inds(A))
        ind_virtual = Index(gensym(:svd))

        U, s, V = tensor_svd_thin(A; inds_u=inds_a_only, ind_s=ind_virtual)

        # absorb singular values
        V = Muscle.hadamard!(V, V, s)

        # contract V against next lane tensor
        V = binary_einsum(B, V)

        # rename back bond index
        U = replace(U, ind_virtual => ind_iso_dir)
        V = replace(V, ind_virtual => ind_iso_dir)

        # replace old tensors with new gauged ones
        psi[i] = U
        psi[i + 1] = V

        # unsafe set of canonical form
        unsafe_setform!(psi, MixedCanonical(site"$(i + 1)"))

        entropies[i] = -sum(x -> x^2 * 2log(x), parent(s))
    end

    return entropies
end

"""
    schmidt_values(psi, bond)

Calculate the Schmidt values of an MPS `psi` at the specified `bond`.

See also: [`schmidt_values!`](@ref).
"""
schmidt_values(psi, bond) = schmidt_values!(copy(psi), bond)

"""
    schmidt_values!(psi, bond)

Calculate the Schmidt values of an MPS `psi` at the specified `bond`.

!!! note

    This function is marked as `!` (in-place) because it modifies the gauge of the MPS during the calculation.

!!! warning

    The MPS should be normalized before calling this function. The function does not normalize the MPS internally.

See also: [`schmidt_values`](@ref).
"""
function schmidt_values!(psi::MPS, bond)
    # TODO use `max` if `bond` is past the middle of the MPS
    _site = min(bond...)
    canonize!(psi, MixedCanonical(site"_site"))

    A = psi[_site]
    B = psi[_site + 1]

    ind_iso_dir = only(intersect(inds(A), inds(B)))
    inds_a_only = filter(!=(ind_iso_dir), inds(A))

    # TODO call eigvals (implement it in Muscle.jl)
    _, s, _ = tensor_svd_thin(A; inds_u=inds_a_only)
    return parent(s)
end

"""
    entropy_vonneumann!(psi, bond)

Calculate the Von Neumann entropy of an MPS `psi` at the specified `bond`.
"""
entropy_vonneumann!(psi, _bond) = -sum(x -> x^2 * 2log(x), schmidt_values!(psi, _bond))

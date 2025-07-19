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
function entropy_vonneumann!(psi)
    entropies = zeros(Float64, nbonds(psi))
    for i in 1:(nsites(psi) - 1)
        entropies[i] = entropy_vonneumann!(psi, bond"$i-$(i + 1)")
    end
    return entropies
end

entropy_vonneumann!(psi, _bond) = -sum(x -> x^2 * 2log(x), schmidt_values!(psi, _bond))

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
    canonize!(psi, bond)
    return parent(tensor_at(psi, LambdaSite(bond)))
end

"""
    entropy_vonneumann!(psi, bond)

Calculate the Von Neumann entropy of an MPS `psi` at the specified `bond`.
"""
entropy_vonneumann!(psi, _bond) = -sum(x -> x^2 * 2log(x), schmidt_values!(psi, _bond))

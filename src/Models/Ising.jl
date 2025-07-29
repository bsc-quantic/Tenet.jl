using ..Tenet

"""
    ising_1d_mpo(L, h, J)

Construct a [`MPO`](@ref) that represents the uniform 1D Ising model with open boundary conditions.

``H(\\sigma) = - J \\sum_{\\langle i, j \\rangle} \\sigma_i \\sigma_j - h \\sum_i \\sigma_i``

where ``\\sigma_i`` is the Pauli-Z operator at site ``i``, ``J`` is the coupling constant, and ``h`` is the external magnetic field.
"""
function ising_1d_mpo(L::Integer, h::Real, J::Real)
    Id = [1.0 0.0; 0.0 1.0]
    sx = [0.0 1.0; 1.0 0.0]
    sz = [1.0 0.0; 0.0 -1.0]

    W_data = zeros(ComplexF64, 2, 2, 3, 3)
    W_data[:, :, 1, 1] .= Id
    W_data[:, :, 2, 1] .= sz
    W_data[:, :, 3, 1] .= -h * sx
    W_data[:, :, 3, 2] .= -J * sz
    W_data[:, :, 3, 3] .= Id

    W_1 = W_data[:, :, 3, :]
    W_n = W_data[:, :, :, 1]

    MPO_data = [W_1, [W_data for i in 2:(L - 1)]..., W_n]

    return MPO(MPO_data; order=(:i, :o, :l, :r))
end

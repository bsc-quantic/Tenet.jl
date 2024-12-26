module TenetKrylovKitExt

using Tenet
using KrylovKit

function eigsolve_prehook_tensor_reshape(A::Tensor, left_inds, right_inds)
    left_inds, right_inds = Tenet.factorinds(A, left_inds, right_inds)

    # Determine the left and right indices
    left_sizes = size.((A,), left_inds)
    right_sizes = size.((A,), right_inds)
    prod_left_sizes = prod(left_sizes)
    prod_right_sizes = prod(right_sizes)

    if prod_left_sizes != prod_right_sizes
        throw(
            ArgumentError("The resulting matrix must be square, but got sizes $prod_left_sizes and $prod_right_sizes.")
        )
    end

    # Permute and reshape the tensor
    A = permutedims(A, [left_inds..., right_inds...])
    Amat = reshape(parent(A), prod_left_sizes, prod_right_sizes)

    return Amat, left_sizes, right_sizes
end

function eigsolve_prehook_tensor_reshape(A::Tensor, x₀::Tensor, left_inds, right_inds)
    left_inds, right_inds = Tenet.factorinds(A, left_inds, right_inds)
    left_inds, right_inds = Tuple(left_inds), Tuple(right_inds)

    Amat, left_sizes, right_sizes = eigsolve_prehook_tensor_reshape(A, left_inds, right_inds)
    prod_left_sizes = prod(left_sizes)

    inds(x₀) != left_inds && throw(
        ArgumentError(
            "The initial guess must have the same left indices as the tensor, but got $(inds(x₀)) and $left_inds."
        ),
    )
    prod(size.((x₀,), left_inds)) != prod_left_sizes && throw(
        ArgumentError(
            "The initial guess must have the same size as the left indices, but got sizes $prod_x₀_sizes and $prod_left_sizes.",
        ),
    )

    # Permute and reshape the tensor
    x₀ = permutedims(x₀, left_inds)
    x₀vec = reshape(parent(x₀), prod_left_sizes)

    return Amat, left_sizes, right_sizes, x₀vec
end

function KrylovKit.eigsolve(
    A::Tensor, howmany::Int=1, which::KrylovKit.Selector=:LM, T::Type=eltype(A); left_inds=(), right_inds=(), kwargs...
)
    Amat, left_sizes, right_sizes = eigsolve_prehook_tensor_reshape(A, left_inds, right_inds)

    # Compute eigenvalues and eigenvectors
    vals, vecs, info = KrylovKit.eigsolve(Amat, howmany, which; kwargs...)

    # Tensorify the eigenvectors
    Avecs = [Tensor(reshape(vec, left_sizes...), left_inds) for vec in vecs]

    return vals, Avecs, info
end

"""
    KrylovKit.eigsolve(tensor::Tensor; left_inds, right_inds, kwargs...)

Perform eigenvalue decomposition on a tensor.

# Keyword arguments

  - `left_inds`: left indices to be used in the eigenvalue decomposition. Defaults to all indices of `t` except `right_inds`.
  - `right_inds`: right indices to be used in the eigenvalue decomposition. Defaults to all indices of `t` except `left_inds`.
"""
function KrylovKit.eigsolve(
    A::Tensor, x₀, howmany::Int, which::KrylovKit.Selector, alg::Algorithm; left_inds=(), right_inds=(), kwargs...
) where {Algorithm<:KrylovKit.Lanczos} # KrylovKit.KrylovAlgorithm}
    Amat, left_sizes, right_sizes = eigsolve_prehook_tensor_reshape(A, left_inds, right_inds)

    # Compute eigenvalues and eigenvectors
    vals, vecs, info = KrylovKit.eigsolve(Amat, x₀, howmany, which, alg; kwargs...)

    # Tensorify the eigenvectors
    Avecs = [Tensor(reshape(vec, left_sizes...), left_inds) for vec in vecs]

    return vals, Avecs, info
end

function KrylovKit.eigsolve(
    A::Tensor,
    x₀::Tensor,
    howmany::Int,
    which::KrylovKit.Selector,
    alg::Algorithm;
    left_inds=inds(x₀),
    right_inds=(),
    kwargs...,
) where {Algorithm<:KrylovKit.Lanczos} # KrylovKit.KrylovAlgorithm}
    Amat, left_sizes, right_sizes, x₀vec = eigsolve_prehook_tensor_reshape(A, x₀, left_inds, right_inds)

    # Compute eigenvalues and eigenvectors
    vals, vecs, info = KrylovKit.eigsolve(Amat, x₀vec, howmany, which, alg; kwargs...)

    # Tensorify the eigenvectors
    Avecs = [Tensor(reshape(vec, left_sizes...), left_inds) for vec in vecs]

    return vals, Avecs, info
end

end

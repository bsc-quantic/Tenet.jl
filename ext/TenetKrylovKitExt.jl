using KrylovKit

"""
    KrylovKit.eigsolve(tensor::Tensor; left_inds, right_inds, kwargs...)

Perform eigenvalue decomposition on a tensor.

# Keyword arguments

  - `left_inds`: left indices to be used in the eigenvalue decomposition. Defaults to all indices of `t` except `right_inds`.
  - `right_inds`: right indices to be used in the eigenvalue decomposition. Defaults to all indices of `t` except `left_inds`.
"""
function KrylovKit.eigsolve(tensor::Tensor; left_inds=(), right_inds=())
    eigsolve(tensor; left_inds=left_inds, right_inds=right_inds)
end

function KrylovKit.eigsolve(tensor::Tensor, x₀::Vector{ComplexF64}, howmany::Int64, which::Union{Symbol, EigSorter}, alg::Lanczos; left_inds=(), right_inds=())
    eigsolve(tensor, x₀, howmany, which, alg; left_inds=left_inds, right_inds=right_inds)
end

function eigsolve(tensor::Tensor, args...; left_inds=(), right_inds=())
    # Determine the left and right indices
    left_inds, right_inds = factorinds(tensor, left_inds, right_inds)

    # Ensure that the resulting matrix is square
    left_sizes = map(Base.Fix1(size, tensor), left_inds)
    right_sizes = map(Base.Fix1(size, tensor), right_inds)
    prod_left_sizes = prod(left_sizes)
    prod_right_sizes = prod(right_sizes)

    if prod_left_sizes != prod_right_sizes
        throw(ArgumentError("The resulting matrix must be square, but got sizes $prod_left_sizes and $prod_right_sizes."))
    end

    # Permute and reshape the tensor
    tensor = permutedims(tensor, [left_inds..., right_inds...])
    data = reshape(parent(tensor), prod_left_sizes, prod_right_sizes)

    # Compute eigenvalues and eigenvectors
    vals, vecs = KrylovKit.eigsolve(data, args...)

    # Tensorify the eigenvectors
    tensor_vecs = [Tensor(reshape(vecs[i], left_sizes...), left_inds) for i in 1:length(vecs)]

    return vals, tensor_vecs
end
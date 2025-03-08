using LinearAlgebra: LinearAlgebra
using cuTensorNet: cuTensorNet

# dispatch to correct architecture
svd(A::Tensor; kwargs...) = svd(arch(A), A; kwargs...)

# TODO use low-rank approximations
"""
    Operations.svd(tensor::Tensor; u_inds, v_inds, bond_ind, kwargs...)

Perform SVD factorization on a tensor. Either `u_inds` or `v_inds` must be specified.

# Keyword arguments

  - `u_inds`: left / U indices to be used in the SVD factorization, except for `bond_ind`.
  - `v_inds`: right / right indices to be used in the SVD factorization, except for `bond_ind`.
  - `bond_ind`: name of the virtual bond.
  - `maxdim`: maximum dimension of the virtual bond.
  - `inplace`: If `true`, it will use `A` as workspace variable to save space. Defaults to `false`.
  - `kwargs...`: additional keyword arguments to be passed to `LinearAlgebra.svd`.
"""
function svd(
    ::CPU,
    A::Tensor;
    u_inds::Vector{Symbol},
    v_inds::Vector{Symbol},
    bond_ind::Symbol,
    maxdim=nothing,
    inplace=false,
    kwargs...,
)
    @assert isdisjoint(u_inds, v_inds)
    @assert u_inds ⊂ inds(A)
    @assert v_inds ⊂ inds(A)
    @assert bond_ind ∉ inds(A)

    # permute array
    left_sizes = map(Base.Fix1(size, tensor), u_inds)
    right_sizes = map(Base.Fix1(size, tensor), v_inds)
    tensor = permutedims(tensor, [u_inds..., v_inds...])
    data = reshape(parent(tensor), prod(left_sizes), prod(right_sizes))

    # compute SVD
    U, s, V = if inplace
        svd!(data; kwargs...)
    else
        svd(data; kwargs...)
    end

    # tensorify results
    U = Tensor(reshape(U, left_sizes..., size(U, 2)), [u_inds..., bond_ind])
    s = Tensor(s, [bond_ind])
    Vt = Tensor(reshape(conj(V), right_sizes..., size(V, 2)), [v_inds..., bond_ind])

    # ad-hoc truncation
    # TODO use low-rank approximations
    if !isnothing(maxdim)
        U = view(U, bond_ind => 1:maxdim)
        s = view(s, bond_ind => 1:maxdim)
        Vt = view(Vt, bond_ind => 1:maxdim)
    end

    return U, s, Vt
end

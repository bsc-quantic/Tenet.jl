# using CUDA: CUDA
# using cuTENSOR: cuTENSOR
using cuTensorNet: cuTensorNet

# TODO automatically move to GPU if G are on CPU?
function simple_update(
    A, ind_physical_a, B, ind_physical_b, ind_bond_ab, G, ind_physical_g_a, ind_physical_g_b; kwargs...
)
    arch_a = arch(A)
    arch_b = arch(B)
    arch_g = arch(G)
    @assert arch_a == arch_b == arch_g

    return simple_update(
        arch_a, A, ind_physical_a, B, ind_physical_b, ind_bond_ab, G, ind_physical_g_a, ind_physical_g_b; kwargs...
    )
end

function simple_update(
    ::CPU,
    A::Tensor,
    ind_physical_a::Symbol,
    B::Tensor,
    ind_physical_b::Symbol,
    ind_bond_ab::Symbol,
    G::Tensor,
    ind_physical_g_a::Symbol,
    ind_physical_g_b::Symbol;
    normalize::Bool=false,
    absorb=nothing,
    atol::Float64=0.0,
    rtol::Float64=0.0,
)
    Θ = contract(contract(A, B; dims=[ind_bond_ab]), G; dims=[ind_physical_a, ind_physical_b])

    # TODO use low-rank approximations
    left_inds = setdiff(inds(A), [ind_physical_a, ind_bond_ab]) ∪ [ind_physical_g_a]
    right_inds = setdiff(inds(B), [ind_physical_b, ind_bond_ab]) ∪ [ind_physical_g_b]
    U, S, V = svd(Θ; left_inds, right_inds, bond_ind=ind_bond_ab)

    normalize && LinearAlgebra.normalize!(S)

    # TODO

    return U, S, V
end

# TODO customize SVD algorithm
# TODO configure GPU stream
# TODO do QR before SU to reduce computational cost on A,B with ninds > 3 but not when size(extent) ~ size(rest)
function simple_update(
    ::GPU{NVIDIA},
    A::Tensor,
    ind_physical_a::Symbol,
    B::Tensor,
    ind_physical_b::Symbol,
    ind_bond_ab::Symbol,
    G::Tensor,
    ind_physical_g_a::Symbol,
    ind_physical_g_b::Symbol;
    normalize::Bool=false,
    absorb=nothing,
    atol::Float64=0.0,
    rtol::Float64=0.0,
)
    all_inds = unique(∪(inds(A), inds(B), inds(G)))
    modes_a = [findfirst(==(i), all_inds) for i in inds(A)]
    modes_b = [findfirst(==(i), all_inds) for i in inds(B)]
    modes_g = [findfirst(==(i), all_inds) for i in inds(G)]

    U = similar(A)
    V = similar(B)
    modes_u = copy(modes_a)
    modes_v = copy(modes_b)

    svd_config = cuTensorNet.SVDConfig(;
        abs_cutoff=atol,
        rel_cutoff=rtol,
        s_partition=if isnothing(absorb)
            cuTensorNet.CUTENSORNET_TENSOR_SVD_PARTITION_NONE
        elseif absorb === :a || absorb === :u || absorb === :left
            cuTensorNet.CUTENSORNET_TENSOR_SVD_PARTITION_US
        elseif absorb === :b || absorb === :v || absorb === :right
            cuTensorNet.CUTENSORNET_TENSOR_SVD_PARTITION_SV
        elseif absorb === :ab || absorb === :equal
            cuTensorNet.CUTENSORNET_TENSOR_SVD_PARTITION_UV_EQUAL
        else
            throw(ArgumentError("Invalid value for absorb: $absorb"))
        end,
        s_normalization=if normalize
            cuTensorNet.CUTENSORNET_TENSOR_SVD_NORMALIZATION_L2
        else
            cuTensorNet.CUTENSORNET_TENSOR_SVD_NORMALIZATION_NONE
        end,
    )

    S_data = similar(parent(A), real(eltype(A)), (size(A, ind_bond_ab),))

    # TODO
    _, _, _, svd_info = cuTENSOR.gateSplit!(
        parent(A), modes_a, parent(B), modes_b, parent(G), modes_g, parent(U), modes_u, S_data, parent(V), modes_v;
    )

    S = Tensor(S_data, [ind_bond_ab])

    return U, S, V, svd_info
end

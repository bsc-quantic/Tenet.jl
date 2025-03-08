# using CUDA: CUDA
# using cuTENSOR: cuTENSOR
using cuTensorNet: cuTensorNet

# absorb behavior trait
# used to keep type-inference happy (`DontAbsorb` returns 3 tensors, while the rest return 2)
abstract type AbsorbBehavior end
struct DontAbsorb <: AbsorbBehavior end
struct AbsorbU <: AbsorbBehavior end
struct AbsorbV <: AbsorbBehavior end
struct AbsorbEqually <: AbsorbBehavior end

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
    absorb::AbsorbBehavior=DontAbsorb(),
    atol::Float64=0.0,
    rtol::Float64=0.0,
)
    Θ = contract(contract(A, B; dims=[ind_bond_ab]), G; dims=[ind_physical_a, ind_physical_b])
    Θ = replace(Θ, ind_physical_g_a => ind_physical_a, ind_physical_g_b => ind_physical_b)

    # TODO use low-rank approximations
    u_inds = setdiff(inds(A), [ind_bond_ab])
    v_inds = setdiff(inds(B), [ind_bond_ab])
    U, S, V = svd(Θ; u_inds, v_inds, s_ind=ind_bond_ab)

    normalize && LinearAlgebra.normalize!(S)

    if absorb isa DontAbsorb
        return U, S, V
    elseif absorb isa AbsorbU
        U = contract(U, S; dims=[])
    elseif absorb isa AbsorbV
        V = contract(V, S; dims=[])
    elseif absorb isa AbsorbEqually
        S_sqrt = sqrt.(S)
        U = contract(U, S_sqrt; dims=[])
        V = contract(V, S_sqrt; dims=[])
    end
    return U, V
end

# TODO customize SVD algorithm
# TODO configure GPU stream
# TODO cache workspace memory
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
    absorb::AbsorbBehavior=DontAbsorb(),
    atol::Float64=0.0,
    rtol::Float64=0.0,
)
    all_inds = unique(∪(inds(A), inds(B), inds(G)))
    modes_a = [findfirst(==(i), all_inds) for i in inds(A)]
    modes_b = [findfirst(==(i), all_inds) for i in inds(B)]
    modes_g = [findfirst(==(i), all_inds) for i in inds(G)]

    U = similar(A)
    V = similar(B)

    # cuTensorNet doesn't like to reuse the physical indices of a and b, so we rename them here
    U = replace(U, ind_physical_a => ind_physical_g_a)
    V = replace(V, ind_physical_b => ind_physical_g_b)

    modes_u = [findfirst(==(i), all_inds) for i in inds(U)]
    modes_v = [findfirst(==(i), all_inds) for i in inds(V)]

    svd_config = cuTensorNet.SVDConfig(;
        abs_cutoff=atol,
        rel_cutoff=rtol,
        s_partition=if absorb isa DontAbsorb
            cuTensorNet.CUTENSORNET_TENSOR_SVD_PARTITION_NONE
        elseif absorb isa AbsorbU
            cuTensorNet.CUTENSORNET_TENSOR_SVD_PARTITION_US
        elseif absorb isa AbsorbV
            cuTensorNet.CUTENSORNET_TENSOR_SVD_PARTITION_SV
        elseif absorb isa AbsorbEqually
            cuTensorNet.CUTENSORNET_TENSOR_SVD_PARTITION_UV_EQUAL
        else
            throw(ArgumentError("Unknown value for absorb: $absorb"))
        end,
        s_normalization=if normalize
            cuTensorNet.CUTENSORNET_TENSOR_SVD_NORMALIZATION_L2
        else
            cuTensorNet.CUTENSORNET_TENSOR_SVD_NORMALIZATION_NONE
        end,
    )

    S_data = similar(parent(A), real(eltype(A)), (size(A, ind_bond_ab),))

    # TODO use svd_info
    _, _, _, svd_info = cuTensorNet.gateSplit!(
        parent(A),
        modes_a,
        parent(B),
        modes_b,
        parent(G),
        modes_g,
        parent(U),
        modes_u,
        S_data,
        parent(V),
        modes_v;
        svd_config,
    )

    S = Tensor(S_data, [ind_bond_ab])

    # undo the index rename to keep cuTensorNet happy
    U = replace(U, ind_physical_g_a => ind_physical_a)
    V = replace(V, ind_physical_g_b => ind_physical_b)

    if absorb isa DontAbsorb
        return U, S, V
    else
        return U, V
    end
end

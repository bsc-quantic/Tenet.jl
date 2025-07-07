module DMRG

using QuantumTags
using Muscle

abstract type Algorithm end

# 1-site DMRG algorithm
struct Dmrg1 <: Algorithm end

# 2-site DMRG algorithm
struct Dmrg2 <: Algorithm end

"""
    dmrg!(ket, ham)

Perform the DMRG algorithm on the given `ket` state and Hamiltonian `ham`.

!!! warning

    This function is still under development, so its API may change in the future as we add more features.
"""
function dmrg! end

struct EffectiveHamiltonian
    le::Tensor
    re::Tensor
    op::Tensor
end

env_left(ham::EffectiveHamiltonian) = ham.le
env_right(ham::EffectiveHamiltonian) = ham.re
op(ham::EffectiveHamiltonian) = ham.op

function (ham::EffectiveHamiltonian)(ψ::Tensor)
    return binary_einsum(binary_einsum(binary_einsum(ham.le, ψ), ham.op), ham.re)
end

struct DmrgState{Ket,Operator}
    ket::Ket
    op::Operator
    envs::Dict{Bond,Tensor}
end

function DmrgState(ket::Ket, op::Operator) where {Ket,Operator} end

function sweep(f, config::DmrgState; dir=:rightleft)
    if dir == :rightleft
        for site in reverse(1:length(config.ket))
            f(config, site)
        end
    elseif dir == :leftright
        for site in 1:length(config.ket)
            f(config, site)
        end
    else
        error("Direction must be either :rightleft or :leftright")
    end
end

hasenv(config::DmrgState, bond::Bond) = haskey(config.envs, bond)
env(config::DmrgState, bond::Bond) = config.envs[bond]
function setenv!(config::DmrgState, bond::Bond, tensor::Tensor)
    config.envs[bond] = tensor
end

getenv(config::DMRGState, site::Site, dir) = get(config.envs, (site, dir), Tensor(fill(1.0)))

function effective_hamiltonian(config::DMRGState, site)
    left_env = getenv(config, CartesianSite(only(site.id) - 1), :left)
    right_env = getenv(config, CartesianSite(only(site.id) + 1), :right)
    op = tensor_at(config.op, site)
    return EffectiveHamiltonian(left_env, right_env, op)
end

function dmrg!(ket, ham)
    # initialize the DMRG state
    config = DmrgState(ket, ham.op)

    # TODO
end

function dmrg_1site!(config::DmrgState)
    # TODO

    energy = ComplexF64(0.0)
    for sweep_it in 1:2
        for i in Iterators.flatten([1:N, N:-1:1])
            left_env = if i == 1
                Tensor(fill(1.0), Index[])
            else
                left_envs[CartesianSite(i - 1)]
            end

            right_env = if i == N
                Tensor(fill(1.0), Index[])
            else
                right_envs[CartesianSite(i + 1)]
            end

            Uold = tensor_at(ψ, CartesianSite(i))
            Vold = tensor_at(ψbra, CartesianSite(i))
            Htensor = tensor_at(H, CartesianSite(i))

            O = binary_einsum(binary_einsum(left_env, Htensor), right_env)

            # @show inds(O) i inds(Uold) inds(Vold)

            U, S, V = tensor_svd_thin(O; inds_u=inds(Uold), inds_v=inds(Vold), ind_s=Index(:tmp))
            Uproj = selectdim(U, Index(:tmp), 1)
            Vproj = selectdim(V, Index(:tmp), 1)

            energy = binary_einsum(binary_einsum(Uproj, O), Vproj)
            @info "$i" energy

            replace_tensor!(ψ, Uold, Uproj)
            replace_tensor!(ψbra, Vold, Vproj)

            left_envs[site"i"] = let
                tmp = binary_einsum(binary_einsum(Uproj, Htensor), Vproj)
                if i != 1
                    tmp = binary_einsum(left_envs[CartesianSite(i - 1)], tmp)
                end
                tmp
            end

            right_envs[site"i"] = let
                tmp = binary_einsum(binary_einsum(Uproj, Htensor), Vproj)
                if i != N
                    tmp = binary_einsum(right_envs[CartesianSite(i + 1)], tmp)
                end
                tmp
            end
        end
    end

    # TODO
end

end
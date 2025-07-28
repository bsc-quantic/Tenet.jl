module DMRG

using ..Tenet
using ..Tenet: ncart_sites, sweep
using QuantumTags
using Muscle
using KrylovKit

# helper types
struct EffectiveHamiltonian
    le::Tensor
    re::Tensor
    op::Tensor
end

function (ham::EffectiveHamiltonian)(ψ::Tensor)
    return binary_einsum(binary_einsum(binary_einsum(ham.le, ψ), ham.op), ham.re)
end

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
export dmrg!

function dmrg!(::Dmrg1, ψ::MPS, op::MPO, nsweeps=4; ishermitian=true, krylovdim=3, maxiter=2, kwargs...)
    energy = ComplexF64(0.0)
    n = ncart_sites(ψ)

    # construct expectation tensor network
    # NOTE ψbra is only required for the indices: it's not updated
    # TODO we should try a better method as it will store old ψ on memory
    ψbra = adjoint_plugs!(copy(ψ))
    resetinds!(ψ)
    resetinds!(op)
    resetinds!(ψbra)
    @align! outputs(ψ) => inputs(op)
    @align! outputs(op) => inputs(ψbra)

    # construct environment on the boundaries as scalar 1 to avoid problems
    envs = Dict{Bond,Tensor}()
    envs[bond"0 - 1"] = Tensor(fill(1.0), Index[])
    envs[bond"$n - $(n + 1)"] = Tensor(fill(1.0), Index[])

    # construct right environments
    for _site in Iterators.drop(sweep(ψ, :left), 1)
        i = only(Tuple(_site))
        (i >= n || i < 1) && continue

        canonize!(ψ, site"$i")

        #! format: off
        envs[bond"$i - $(i+1)"] = binary_einsum(
            binary_einsum(
                op[site"$i+1"],
                binary_einsum(
                    ψ[site"$i+1"],
                    envs[bond"$(i+1) - $(i+2)"]
                )
            ),
            conj(ψbra[site"$i+1"])
        )
        #! format: on
    end

    for sweep_it in 1:nsweeps
        # left-to-right sweep
        # NOTE skip first iteration as it will be computed by the right-to-left sweep
        for _site in Iterators.drop(sweep(ψ, :right), 1)
            i = only(Tuple(_site))

            # move orthogonality center to the site
            canonize!(ψ, _site)

            # update left environments on the go
            ψbra_oldsite_tensor = replace(
                conj(ψ[site"$(i-1)"]),
                ind_at(ψ, plug"$(i-1)") => ind_at(ψbra, plug"$(i-1)'"),
                ind_at(ψ, bond"$(i-1) - $i") => ind_at(ψbra, bond"$(i-1) - $i"),
            )
            if _site != site"2"
                ψbra_oldsite_tensor = replace(
                    ψbra_oldsite_tensor, ind_at(ψ, bond"$(i-2) - $(i-1)") => ind_at(ψbra, bond"$(i-2) - $(i-1)")
                )
            end
            #! format: off
            envs[bond"$(i-1) - $i"] = binary_einsum(
                binary_einsum(
                    binary_einsum(
                        envs[bond"$(i-2) - $(i-1)"],
                        ψ[site"$(i-1)"]
                    ),
                    op[site"$(i-1)"]
                ),
                ψbra_oldsite_tensor
            )
            #! format: on

            # construct effective hamiltonian and solve for the ground state
            Heff = EffectiveHamiltonian(envs[bond"$(i-1) - $i"], envs[bond"$i - $(i+1)"], op[site"$i"])
            xθ = ψ[site"$i"]

            # replace indices (`Heff(x)` returns bra indices, but we want ket indices)
            replacements = [ind_at(ψbra, plug"$i'") => ind_at(ψ, plug"$i")]
            if _site != site"1"
                push!(replacements, ind_at(ψbra, bond"$(i-1) - $i") => ind_at(ψ, bond"$(i-1) - $i"))
            end
            if _site != site"$n"
                push!(replacements, ind_at(ψbra, bond"$i - $(i+1)") => ind_at(ψ, bond"$i - $(i+1)"))
            end

            eigvals, eigvecs, info = eigsolve(xθ, 1; ishermitian, krylovdim, maxiter, kwargs...) do x
                # efficiently contract application of x ket vector with effective hamiltonian
                y = Heff(x)

                # replace indices (`Heff(x)` returns bra indices, but we want ket indices)
                y = replace(y, replacements...)

                permutedims(y, replace(inds(xθ), replacements...))
            end

            # update tensor and info
            energy = eigvals[1]
            y = eigvecs[1]

            ψ[_site] = y
        end

        # right-to-left sweep
        # NOTE skip first iteration as it will be computed by the left-to-right sweep
        for _site in Iterators.drop(sweep(ψ, :left), 1)
            i = only(Tuple(_site))

            # move orthogonality center to the site
            canonize!(ψ, _site)

            # update right environments on the go
            ψbra_oldsite_tensor = replace(
                conj(ψ[site"$(i+1)"]),
                ind_at(ψ, plug"$(i+1)") => ind_at(ψbra, plug"$(i+1)'"),
                ind_at(ψ, bond"$i - $(i+1)") => ind_at(ψbra, bond"$i - $(i+1)"),
            )
            if _site != site"$n-1"
                ψbra_oldsite_tensor = replace(
                    ψbra_oldsite_tensor, ind_at(ψ, bond"$(i+1) - $(i+2)") => ind_at(ψbra, bond"$(i+1) - $(i+2)")
                )
            end
            #! format: off
            envs[bond"$i - $(i+1)"] = binary_einsum(
                binary_einsum(
                    binary_einsum(
                        envs[bond"$(i+1) - $(i+2)"],
                        ψ[site"$(i+1)"]
                    ),
                    op[site"$(i+1)"]
                ),
                ψbra_oldsite_tensor
            )
            #! format: on

            # construct effective hamiltonian and solve for the ground state
            Heff = EffectiveHamiltonian(envs[bond"$(i-1) - $i"], envs[bond"$i - $(i+1)"], op[site"$i"])
            xθ = ψ[site"$i"]

            # replace indices (`Heff(x)` returns bra indices, but we want ket indices)
            replacements = [ind_at(ψbra, plug"$i'") => ind_at(ψ, plug"$i")]
            if _site != site"1"
                push!(replacements, ind_at(ψbra, bond"$(i-1) - $i") => ind_at(ψ, bond"$(i-1) - $i"))
            end
            if _site != site"$n"
                push!(replacements, ind_at(ψbra, bond"$i - $(i+1)") => ind_at(ψ, bond"$i - $(i+1)"))
            end

            eigvals, eigvecs, info = eigsolve(xθ, 1; ishermitian, krylovdim, maxiter, kwargs...) do x
                # efficiently contract application of x ket vector with effective hamiltonian
                y = Heff(x)

                # replace indices (`Heff(x)` returns bra indices, but we want ket indices)
                y = replace(y, replacements...)

                permutedims(y, replace(inds(xθ), replacements...))
            end

            # update tensor and info
            energy = eigvals[1]
            y = eigvecs[1]

            ψ[_site] = y
        end
    end

    return canonicalize_inds!(ψ), energy
end

end
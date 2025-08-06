using DelegatorTraits

struct Evolvable <: Interface end

# trait
abstract type EvolveAlgorithm end
struct UnknownAlgorithm <: EvolveAlgorithm end
struct SimpleUpdate <: EvolveAlgorithm end
struct DirectMPOContraction <: EvolveAlgorithm end
struct ZipUpAlgorithm <: EvolveAlgorithm end

function evolve! end

function evolve(tn, op; kwargs...)
    tn = copy(tn)
    evolve!(tn, op; kwargs...)
end

evolve!(tn, op; kwargs...) = evolve!(tn, op, DelegatorTrait(Evolvable(), tn); kwargs...)
evolve!(tn, op, ::DelegateToField; kwargs...) = evolve!(delegator(Evolvable(), tn), op; kwargs...)
evolve!(tn, op, ::DontDelegate; algorithm=UnknownAlgorithm(), kwargs...) = evolve!(algorithm, tn, op; kwargs...)

# TODO use an `Algorithm` trait to dispatch on the algorithm
function generic_evolve_mps_mpo_direct!(mps, op)
    @argcheck nsites(mps) == nsites(op) "MPS and MPO must have the same number of sites"

    # align MPS and MPO
    op = resetinds!(copy(op))
    align!(mps, :outputs, op, :inputs)

    @unsafe_region mps for i in 1:nsites(mps)
        tensor_mps = tensor_at(mps, site"$i")
        tensor_op = tensor_at(op, site"$i")
        c = binary_einsum(tensor_mps, tensor_op)
        c = replace(c, ind_at(op, plug"$i") => ind_at(mps, plug"$i"))

        # fuse virtual indices
        if i > 1
            j = i - 1
            c = Muscle.fuse(c, [ind_at(mps, bond"$j - $i"), ind_at(op, bond"$j - $i")]; ind=ind_at(mps, bond"$j - $i"))
        end
        if i < nsites(mps)
            j = i + 1
            c = Muscle.fuse(c, [ind_at(mps, bond"$i - $j"), ind_at(op, bond"$i - $j")]; ind=ind_at(mps, bond"$i - $j"))
        end

        replace_tensor!(mps, tensor_mps, c)
    end

    return mps
end

# NOTE last bonds use full maxdim size, instead of the one due to boundary effects
function generic_evolve_mps_mpo_zipup!(mps, op; maxdim=nothing, threhold=nothing)
    @argcheck nsites(mps) == nsites(op) "MPS and MPO must have the same number of sites"
    @argcheck !isnothing(maxdim) || !isnothing(threhold) "Either `maxdim` or `threhold` must be specified, or use"

    # align MPS and MPO
    op = resetinds!(copy(op))
    align!(mps, :outputs, op, :inputs)

    C₁ = binary_einsum(tensor_at(mps, site"1"), tensor_at(op, site"1"))
    C₁ = replace(C₁, ind_at(op, plug"1") => ind_at(mps, plug"1"))

    # better use truncated `eigen` or truncated `svd`?
    ind_s = Index(gensym(:tmp))
    U, S, V = tensor_svd_thin(C₁; inds_u=[ind_at(mps, plug"1")], ind_s)
    if !isnothing(maxdim)
        U = view(U, ind_s => 1:min(maxdim, length(S)))
        S = view(S, ind_s => 1:min(maxdim, length(S)))
        V = view(V, ind_s => 1:min(maxdim, length(S)))
    end
    if !isnothing(threhold)
        keep = findall(x -> abs(x) > threhold, S)
        U = view(U, ind_s => keep)
        S = view(S, ind_s => keep)
        V = view(V, ind_s => keep)
    end

    # absorb the singular values into V to shift right the orthogonality center
    R = binary_einsum(V, S; dims=Index[])

    # rename temporal index to the bond index
    U = replace(U, ind_s => ind_at(mps, bond"1-2"))

    @unsafe_region mps begin
        replace_tensor!(mps, tensor_at(mps, site"1"), U)

        for i in 2:(nsites(mps) - 1)
            _site = site"$i"
            R = binary_einsum(binary_einsum(R, tensor_at(mps, _site)), tensor_at(op, _site))
            R = replace(R, ind_at(op, plug"$i") => ind_at(mps, plug"$i"))

            # rename left temporal index to the bond index
            R = replace(R, ind_s => ind_at(mps, bond"$(i - 1) - $i"))

            U, S, V = tensor_svd_thin(R; inds_u=[ind_at(mps, plug"$i"), ind_at(mps, bond"$(i - 1) - $i")], ind_s)

            if !isnothing(maxdim)
                U = view(U, ind_s => 1:min(maxdim, length(S)))
                S = view(S, ind_s => 1:min(maxdim, length(S)))
                V = view(V, ind_s => 1:min(maxdim, length(S)))
            end

            if !isnothing(threhold)
                keep = findall(x -> abs(x) > threhold, S)
                U = view(U, ind_s => keep)
                S = view(S, ind_s => keep)
                V = view(V, ind_s => keep)
            end

            # absorb the singular values into V to shift right the orthogonality center
            R = binary_einsum(V, S; dims=Index[])

            # rename temporal index to the bond index
            U = replace(U, ind_s => ind_at(mps, bond"$i - $(i + 1)"))

            replace_tensor!(mps, tensor_at(mps, _site), U)
        end

        # last site
        i = nsites(mps)
        R = binary_einsum(R, tensor_at(mps, site"$i"))
        R = binary_einsum(R, tensor_at(op, site"$i"))
        R = replace(R, ind_at(op, plug"$i") => ind_at(mps, plug"$i"))

        # rename left temporal index to the bond index
        R = replace(R, ind_s => ind_at(mps, bond"$(i - 1) - $i"))

        replace_tensor!(mps, tensor_at(mps, site"$i"), R)
    end

    return mps
end

function evolve!(mps::MPS, op::AbstractMPO)
    generic_evolve_mps_mpo_direct!(mps, op)

    # direct method loses canonicity
    canonize!(mps, MixedCanonical(sites(mps)))

    return mps
end

function evolve!(tn::AbstractTangle, op::Tensor)
    simple_update!(tn, op)
    return tn
end

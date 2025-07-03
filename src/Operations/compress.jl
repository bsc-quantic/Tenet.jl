using Muscle

function compress! end

function compress!(ψ::MPS; maxdim=nothing, threshold=nothing, kwargs...)
    @argcheck !isnothing(maxdim) || !isnothing(threshold) "Either `maxdim` or `threshold` must be specified."
    @argcheck isnothing(maxdim) || maxdim > 0 "maxdim must be a positive integer."
    @argcheck isnothing(threshold) || threshold > 0 "Threshold must be positive."

    canonize!(ψ, MixedCanonical(site"1"))

    for i in 1:(nsites(ψ) - 1)
        _bond = bond"$i-$(i+1)"
        a = tensor_at(ψ, site"$i")
        b = tensor_at(ψ, site"$(i + 1)")

        ind_tmp = Index(gensym(:tmp))
        U, S, V = Muscle.tensor_svd_thin(a; inds_v=[ind_at(ψ, _bond)], ind_s=ind_tmp)
        if !isnothing(maxdim)
            U = view(U, ind_tmp => 1:min(maxdim, length(S)))
            S = view(S, ind_tmp => 1:min(maxdim, length(S)))
            V = view(V, ind_tmp => 1:min(maxdim, length(S)))
        end

        # use `threshold` to truncate the singular values
        if !isnothing(threshold)
            keep = findall(x -> abs(x) > threshold, S)
            U = view(U, ind_tmp => keep)
            S = view(S, ind_tmp => keep)
            V = view(V, ind_tmp => keep)
        end

        # absorb the singular values into V to shift right the orthogonality center
        V = hadamard!(V, V, S)

        # contract against tensor in next site
        V = binary_einsum(V, b; dims=[ind_at(ψ, _bond)])

        # rename the temporal index to the bond index
        V = replace(V, ind_tmp => ind_at(ψ, _bond))
        U = replace(U, ind_tmp => ind_at(ψ, _bond))

        # update the tensor network
        @unsafe_region ψ begin
            replace_tensor!(ψ, a, U)
            replace_tensor!(ψ, b, V)
        end
        ψ.form = MixedCanonical(site"$(i + 1)")
    end

    return ψ
end

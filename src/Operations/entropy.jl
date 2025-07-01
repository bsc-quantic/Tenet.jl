function vn_entanglement_entropy(psi)
    psi = copy(psi) #deepcopy ? 
    vn_entanglement_entropy!(psi)
end

function vn_entanglement_entropy!(psi)
    normalize!(psi)
    #@info norm(psi)

    N = length(psi)

    canonize!(psi, MixedCanonical(site"1"))

    s_vn = zeros(Float64, N-1)
    for _site in 1:(N - 1)

        # A it the tensor where we perform the factorization, but B is also affected by the gauge transformation
        A = psi[_site]
        B = psi[_site + 1]

        ind_iso_dir = only(intersect(inds(A), inds(B)))
        inds_a_only = filter(!=(ind_iso_dir), inds(A))
        ind_virtual = Index(gensym(:tmp))

        U, s, V = tensor_svd_thin(A; inds_u=inds_a_only, ind_s=ind_virtual)

        # absorb singular values
        V = Muscle.hadamard(V, s)

        # contract V against next lane tensor
        V = binary_einsum(B, V)

        # rename back bond index
        U = replace(U, ind_virtual => ind_iso_dir)
        V = replace(V, ind_virtual => ind_iso_dir)

        # replace old tensors with new gauged ones
        psi[_site] = U
        psi[_site + 1] = V

        psi.form = MixedCanonical(CartesianSite(_site + 1))

        s2 = parent(s) .^ 2
        #@info s2

        s_vn[_site] = -sum(s2 .* log.(s2))
    end

    return s_vn
end

function sergio_values!(psi, cut)
    canonize!(psi, MixedCanonical(CartesianSite(cut)))

        A = psi[cut]
        B = psi[cut+1]

        ind_iso_dir = only(intersect(inds(A), inds(B)))
        inds_a_only = filter(!=(ind_iso_dir), inds(A))

        _, s, _ = tensor_svd_thin(A; inds_u=inds_a_only)
 
    return parent(s) .^ 2
end

function sergio_entropy(lambdas::AbstractVector)
    return -sum(lambdas .* log.(lambdas))
end

function sergio_entropy!(psi, cut)
    sergio_entropy(sergio_values!(psi,cut))
end

function sergio_entropy!(psi::AbstractMPS)
    N = length(psi)
    vn_ent = zeros(Float64, N-1)

    for cut in eachindex(vn_ent)
        vn_ent[cut] = sergio_entropy!(psi,cut)
    end
    return vn_ent
end

function sergio_entropy(psi::AbstractMPS)
    sergio_entropy!(copy(psi))
end
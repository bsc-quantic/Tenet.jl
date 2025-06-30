function vn_entanglement_entropy(psi)
    psi = copy(psi) #deepcopy ? 
    vn_entanglement_entropy!(psi)
end

function vn_entanglement_entropy!(psi)
    normalize!(psi)
    @info norm(psi)

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

        psi.orthog_center = MixedCanonical(CartesianSite(_site + 1))

        s2 = parent(s) .^ 2
        #@info s2

        s_vn[_site] = -sum(s2 .* log.(s2))
    end

    return s_vn
end

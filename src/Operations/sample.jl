using Distributions: Distributions

function sample end

# TODO multisampling by using a batch dimension
# TODO acceleration by grouping sites in blocks?
# based on https://tensornetwork.org/mps/algorithms/sampling/
function sample(ψ::MPS, nsamples=1)
    _samples = [sizehint!(Int[], nsites(ψ)) for _ in 1:nsamples]

    # canonize to first site such that the contraction of right tensors are the identity
    tn = canonize(ψ, MixedCanonical(site"1"))

    for k in 1:nsamples
        proj_tensor = Tensor(fill(1))

        for i in 1:nsites(ψ)
            # compute the marginal probability distribution for the current site
            t = binary_einsum(proj_tensor, tensor_at(tn, site"$i"))
            physind = ind_at(tn, plug"$i")
            marg_prob_dist = vec(Base._mapreduce_dim(abs2, +, zero(eltype(t)), t, filter(!=(physind), inds(t))))

            # randomly sample a value from the marginal distribution
            chosen_value = rand(Distributions.Categorical(marg_prob_dist))
            push!(_samples[k], chosen_value)

            # update the projection tensor that informs of the of the already sampled values for correlation correction
            proj_tensor = view(t, physind => chosen_value) / sqrt(marg_prob_dist[chosen_value])
        end
    end

    return _samples
end

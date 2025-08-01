using Distributions: Distributions

"""
    sample(tn, nsamples=1; kwargs...)

Sample `nsamples` samples from a Tensor Network `tn`.

The samples are returned as a `Vector{Vector{Int}}` (i.e. a list of "bitstrings"), where each number corresponds to the index of the sampled local physical state.
For example, if working on a 2-dim physical space, the sampled values will be `1` or `2`, corresponding to the ``\\ket{0}`` and ``\\ket{1}`` states, respectively.

!!! warning

    The interface is still experimental and may change in the future.
"""
function sample end

# TODO multisampling by using a batch dimension
# TODO acceleration by grouping sites in blocks?
# based on https://tensornetwork.org/mps/algorithms/sampling/
function sample(ψ::MPS, nsamples=1; batchdim=4)
    # if nsamples is less than batchdim, it doesn't make much sense to use a bigger batch dimension
    batchdim = min(batchdim, nsamples)

    nsamples = batchdim * ceil(Int, nsamples / batchdim)
    samples = [zeros(Int, nsites(ψ)) for _ in 1:nsamples]
    batchind = Index(gensym(:batch))

    # canonize to first site such that the contraction of right tensors are the identity
    tn = canonize(ψ, MixedCanonical(site"1"))

    for k_batch in 0:(ceil(Int, nsamples / batchdim) - 1)
        proj_tensor = Tensor(ones(Int, batchdim), [batchind])

        for i in 1:nsites(ψ)
            # compute the marginal probability distribution for the current site
            t = binary_einsum(proj_tensor, tensor_at(tn, site"$i"))
            physind = ind_at(tn, plug"$i")

            if i != nsites(ψ)
                bondind = only(filter(x -> x != physind && x != batchind, inds(t)))
                proj_tensor = Tensor(zeros(eltype(t), batchdim, size(t, bondind)), [batchind, bondind])
            end

            # randomly sample a value from the marginal distribution
            for k_inner in 1:batchdim
                k_total = k_batch * batchdim + k_inner
                marg_prob_dist = if i == nsites(ψ)
                    abs2.(parent(view(t, batchind => k_inner)))
                else
                    t_inner = view(t, batchind => k_inner)
                    vec(Base._mapreduce_dim(abs2, +, zero(eltype(t)), t_inner, filter(!=(physind), inds(t_inner))))
                end

                # ensure it's real for Categorical distribution
                @assert isreal(marg_prob_dist)
                marg_prob_dist = real(marg_prob_dist)

                chosen_value = rand(Distributions.Categorical(marg_prob_dist))
                samples[k_total][i] = chosen_value

                # update the projection tensor that informs of the of the already sampled values for correlation correction
                if i != nsites(ψ)
                    view(proj_tensor, batchind => k_inner) .=
                        view(view(t, batchind => k_inner), physind => chosen_value) ./
                        sqrt(marg_prob_dist[chosen_value])
                end
            end
        end
    end

    return samples
end

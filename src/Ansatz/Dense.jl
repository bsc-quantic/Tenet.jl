struct Dense <: Ansatz
    super::Quantum
end

function Dense(::State, array::AbstractArray; sites=Site.(1:ndims(array)))
    @assert ndims(array) > 0
    @assert all(>(1), size(array))

    gen = IndexCounter()
    symbols = [nextindex!(gen) for _ in 1:ndims(array)]
    sitemap = Dict{Site,Symbol}(
        map(sites, 1:ndims(array)) do site, i
            site => symbols[i]
        end,
    )

    tensor = Tensor(array, symbols)

    tn = TensorNetwork([tensor])
    qtn = Quantum(tn, sitemap)
    return Dense(qtn)
end

function Dense(::Operator, array::AbstractArray; sites)
    @assert ndims(array) > 0
    @assert all(>(1), size(array))
    @assert length(sites) == ndims(array)

    gen = IndexCounter()
    tensor_inds = [nextindex!(gen) for _ in 1:ndims(array)]
    tensor = Tensor(array, tensor_inds)
    tn = TensorNetwork([tensor])

    sitemap = Dict{Site,Symbol}(map(splat(Pair), zip(sites, tensor_inds)))
    qtn = Quantum(tn, sitemap)

    return Dense(qtn)
end

Base.copy(qtn::Dense) = Dense(copy(Quantum(qtn)))

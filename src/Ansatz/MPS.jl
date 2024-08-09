struct MPS <: Ansatz
    super::Quantum
end

function MPS(arrays::Vector{<:AbstractArray}; order=defaultorder(MPS))
    @assert ndims(arrays[1]) == 2 "First array must have 2 dimensions"
    @assert all(==(3) ∘ ndims, arrays[2:(end - 1)]) "All arrays must have 3 dimensions"
    @assert ndims(arrays[end]) == 2 "Last array must have 2 dimensions"
    issetequal(order, defaultorder(Chain, State())) ||
        throw(ArgumentError("order must be a permutation of $(String.(defaultorder(Chain, State())))"))

    n = length(arrays)
    gen = IndexCounter()
    symbols = [nextindex!(gen) for _ in 1:(2n)]

    _tensors = map(enumerate(arrays)) do (i, array)
        _order = if i == 1
            filter(x -> x != :l, order)
        elseif i == n
            filter(x -> x != :r, order)
        else
            order
        end

        inds = map(_order) do dir
            if dir == :o
                symbols[i]
            elseif dir == :r
                symbols[n + mod1(i, n)]
            elseif dir == :l
                symbols[n + mod1(i - 1, n)]
            else
                throw(ArgumentError("Invalid direction: $dir"))
            end
        end
        Tensor(array, inds)
    end

    sitemap = Dict(Site(i) => symbols[i] for i in 1:n)

    return MPS(Quantum(TensorNetwork(_tensors), sitemap))
end

boundary(::MPS) = Open()

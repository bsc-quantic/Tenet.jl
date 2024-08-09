struct MPO <: Ansatz
    super::Quantum
end

function MPO(arrays::Vector{<:AbstractArray}; order=defaultorder(MPO))
    @assert ndims(arrays[1]) == 3 "First array must have 3 dimensions"
    @assert all(==(4) ∘ ndims, arrays[2:(end - 1)]) "All arrays must have 4 dimensions"
    @assert ndims(arrays[end]) == 3 "Last array must have 3 dimensions"
    issetequal(order, defaultorder(Chain, Operator())) ||
        throw(ArgumentError("order must be a permutation of $(String.(defaultorder(Chain, Operator())))"))

    n = length(arrays)
    gen = IndexCounter()
    symbols = [nextindex!(gen) for _ in 1:(3n - 1)]

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
            elseif dir == :i
                symbols[i + n]
            elseif dir == :l
                symbols[2n + mod1(i - 1, n)]
            elseif dir == :r
                symbols[2n + mod1(i, n)]
            else
                throw(ArgumentError("Invalid direction: $dir"))
            end
        end
        Tensor(array, inds)
    end

    sitemap = Dict(Site(i) => symbols[i] for i in 1:n)
    merge!(sitemap, Dict(Site(i; dual=true) => symbols[i + n] for i in 1:n))

    return MPO(Quantum(TensorNetwork(_tensors), sitemap))
end

boundary(::MPO) = Open()

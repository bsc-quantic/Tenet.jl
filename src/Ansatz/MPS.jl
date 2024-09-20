using Random

abstract type AbstractMPS <: Ansatz end

struct MPS <: AbstractMPS
    super::Quantum
    form::Form
end

defaultorder(::Type{MPS}) = (:o, :l, :r)
boundary(::MPS) = Open()
form(tn::MPS) = tn.form

function MPS(arrays::Vector{<:AbstractArray}; order=defaultorder(MPS))
    @assert ndims(arrays[1]) == 2 "First array must have 2 dimensions"
    @assert all(==(3) ∘ ndims, arrays[2:(end - 1)]) "All arrays must have 3 dimensions"
    @assert ndims(arrays[end]) == 2 "Last array must have 2 dimensions"
    issetequal(order, defaultorder(MPS)) ||
        throw(ArgumentError("order must be a permutation of $(String.(defaultorder(MPS)))"))

    n = length(arrays)
    gen = IndexCounter()
    symbols = [nextindex!(gen) for _ in 1:(2n)]

    tn = TensorNetwork(
        map(enumerate(arrays)) do (i, array)
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
        end,
    )

    sitemap = Dict(Site(i) => symbols[i] for i in 1:n)
    qtn = Quantum(tn, sitemap)

    return MPS(qtn, NonCanonical())
end

function Base.convert(::Type{MPS}, tn::Product)
    @assert socket(tn) == State()

    arrs::Vector{Array} = arrays(tn)
    arrs[1] = reshape(arrs[1], size(arrs[1])..., 1)
    arrs[end] = reshape(arrs[end], size(arrs[end])..., 1)
    map!(@view(arrs[2:(end - 1)]), @view(arrs[2:(end - 1)])) do arr
        reshape(arr, size(arr)..., 1, 1)
    end

    return MPS(arrs)
end

Base.adjoint(tn::MPS) = MPS(adjoint(Quantum(tn)), form(tn))

# TODO different input/output physical dims
# TODO let choose the orthogonality center
function Base.rand(rng::Random.AbstractRNG, ::Type{MPS}, n, χ; eltype=Float64, physical_dim=2)
    p = physical_dim
    T = eltype

    arrays::Vector{AbstractArray{T,N} where {N}} = map(1:n) do i
        χl, χr = let after_mid = i > n ÷ 2, i = (n + 1 - abs(2i - n - 1)) ÷ 2
            χl = min(χ, p^(i - 1))
            χr = min(χ, p^i)

            # swap bond dims after mid and handle midpoint for odd-length MPS
            (isodd(n) && i == n ÷ 2 + 1) ? (χl, χl) : (after_mid ? (χr, χl) : (χl, χr))
        end

        # orthogonalize by QR factorization
        F = lq!(rand(rng, T, χl, p * χr))

        reshape(Matrix(F.Q), χl, p, χr)
    end

    # reshape boundary sites
    arrays[1] = reshape(arrays[1], p, p)
    arrays[n] = reshape(arrays[n], p, p)

    return MPS(arrays; order=(:l, :o, :r))
end

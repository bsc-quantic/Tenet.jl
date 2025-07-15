using DelegatorTraits
using Tangles
using Random

abstract type AbstractMPO <: AbstractTangle end

defaultorder(::Type{<:AbstractMPO}) = (:l, :r, :o, :i)

"""
    MatrixProductOperator

A Matrix Product Operator (MPO) Tensor Network.
"""
struct MatrixProductOperator <: AbstractMPO
    tn::GenericTensorNetwork
end

const MPO = MatrixProductOperator

Base.copy(tn::MPO) = MPO(copy(tn.tn))

ImplementorTrait(interface, tn::MPO) = ImplementorTrait(interface, tn.tn)
function DelegatorTrait(interface, tn::MPO)
    if ImplementorTrait(interface, tn.tn) === Implements()
        DelegateToField{:tn}()
    else
        DontDelegate()
    end
end

function MPO(arrays::Vector; order=defaultorder(MPO))
    @assert ndims(arrays[1]) == 3 "First array must have 3 dimensions"
    @assert all(==(4) ∘ ndims, arrays[2:(end - 1)]) "All arrays must have 4 dimensions"
    @assert ndims(arrays[end]) == 3 "Last array must have 3 dimensions"
    issetequal(order, defaultorder(MPO)) ||
        throw(ArgumentError("order must be a permutation of $(String.(defaultorder(MPO)))"))

    # n = length(arrays)
    # gen = IndexCounter()
    # lattice = Lattice(Val(:chain), n)

    # sitemap = Dict{Site,Symbol}(Site(i) => nextindex!(gen) for i in 1:n)
    # merge!(sitemap, Dict([Site(i; dual=true) => nextindex!(gen) for i in 1:n]))
    # bondmap = Dict{Bond,Symbol}(bond => nextindex!(gen) for bond in Graphs.edges(lattice))

    tn = GenericTensorNetwork()

    for (i, array) in enumerate(arrays)
        isub = i - 1
        isup = i + 1

        local_order = if i == 1
            filter(x -> x != :l, order)
        elseif i == length(arrays)
            filter(x -> x != :r, order)
        else
            order
        end

        inds = map(local_order) do dir
            if dir == :o
                Index(plug"$i")
            elseif dir == :i
                Index(plug"$i'")
            elseif dir == :r
                Index(bond"$i-$isup")
            elseif dir == :l
                Index(bond"$isub-$i")
            else
                throw(ArgumentError("Invalid direction: $dir"))
            end
        end |> collect

        _tensor = Tensor(array, inds)
        addtensor!(tn, _tensor)
        setsite!(tn, _tensor, site"$i")
        Tangles.setplug!(tn, Index(plug"$i"), plug"$i")
        Tangles.setplug!(tn, Index(plug"$i'"), plug"$i'")
        hasbond(tn, bond"$i-$isup") ||
            hasind(tn, Index(bond"$i-$isup")) && setbond!(tn, Index(bond"$i-$isup"), bond"$i-$isup")
        hasbond(tn, bond"$isub-$i") ||
            hasind(tn, Index(bond"$isub-$i")) && setbond!(tn, Index(bond"$isub-$i"), bond"$isub-$i")
    end

    return MPO(tn)
end

# CanonicalForm trait
CanonicalForm(::MPO) = NonCanonical()

# TODO normalize as we canonize for numerical stability
# TODO different input/output physical dims
# TODO let choose the orthogonality center
"""
    Base.rand(rng::Random.AbstractRNG, ::Type{MPO}; n, maxdim, eltype=Float64, physdim=2)

Create a random [`MPO`](@ref) Tensor Network.
In order to avoid norm explosion issues, the tensors are orthogonalized by QR factorization so its normalized and mixed canonized to the last site.

# Keyword Arguments

  - `n` The number of sites.
  - `maxdim` The maximum bond dimension. If it is `nothing`, the maximum bond dimension increases exponentially with the number of sites up to `(physdim^2)^(n ÷ 2)`.
  - `eltype` The element type of the tensors. Defaults to `Float64`.
  - `physdim` The physical or output dimension of each site. Defaults to 2.
"""
function Base.rand(rng::Random.AbstractRNG, ::Type{MPO}; n, maxdim=nothing, eltype=Float64, physdim=2)
    T = eltype
    ip = op = physdim
    χ = isnothing(maxdim) ? Base.Checked.checked_pow(Base.Checked.checked_mul(ip * op), n ÷ 2) : maxdim

    arrays::Vector{AbstractArray{T,N} where {N}} = map(1:n) do i
        χl, χr = let after_mid = i > n ÷ 2, i = (n + 1 - abs(2i - n - 1)) ÷ 2
            χl = min(χ, ip^(i - 1) * op^(i - 1))
            χr = min(χ, ip^i * op^i)

            χl = min(
                χ,
                try
                    a = Base.Checked.checked_pow(ip, i - 1)
                    b = Base.Checked.checked_pow(op, i - 1)
                    Base.Checked.checked_mul(a, b)
                catch e
                    if e isa OverflowError
                        typemax(Int)
                    else
                        rethrow(e)
                    end
                end,
            )

            χr = min(
                χ,
                try
                    a = Base.Checked.checked_pow(ip, i)
                    b = Base.Checked.checked_pow(op, i)
                    Base.Checked.checked_mul(a, b)
                catch e
                    if e isa OverflowError
                        typemax(Int)
                    else
                        rethrow(e)
                    end
                end,
            )

            # swap bond dims after mid and handle midpoint for odd-length MPO
            (isodd(n) && i == n ÷ 2 + 1) ? (χl, χl) : (after_mid ? (χr, χl) : (χl, χr))
        end

        # orthogonalize by QR factorization
        F = lq!(rand(rng, T, χl, ip * op * χr))
        reshape(Matrix(F.Q), χl, ip, op, χr)
    end

    # reshape boundary sites
    arrays[1] = reshape(arrays[1], ip, op, min(χ, ip * op))
    arrays[n] = reshape(arrays[n], min(χ, ip * op), ip, op)

    # TODO order might not be the best for performance
    return MPO(arrays; order=(:l, :i, :o, :r))
end

"""
    autompo_periodic(L, one_body, two_body; eltype=ComplexF64)

Builds a translationally invariant MPO of length ``L`` for Hamiltonians of the form:

 ``H = sum_i c1 * O_i + sum_{i,j} c2 * O_i ⊗ O_j``

where ``O_i`` and ``O_j`` are local operators acting on sites ``i`` and ``j``, respectively.
Several pairs (i,j) can be passed with different coefficients c2 for terms that insert operators at different distances. For example adding a (1,2) term insert nearest-neighbor interactions, while a (1,3) term inserts next-nearest-neighbor interactions.

#Example usage

one_body = [(1, X, 1.0), (1, Y, 0.5)]
two_body = [(1, 2, X, X, 0.1), (1, 3, Z, Z, 0.05)]

H = autompo_periodic(4, one_body, two_body)

Returns the Hamiltonian 
H = ``sum_{i=1}^4 (1.0 * X_i + 0.5 * Y_i) + sum_{i=1}^3  0.1 * X_i ⊗ X_{i+1} + sum_{i=1}^2 0.05 * Z_i ⊗ Z_{i+2})``

# Arguments

  - `L` : length of the MPO
  - `one_body` : list of single‐site terms of the form `(i, Oi, alpha)`
  - `two_body` : list of two‐site terms `(i, j, Oi, Oj, beta)`
"""

function autompo_periodic(L, one_body, two_body; type=ComplexF64)
    loc_dim = size(one_body[1][2])[1]  # Local physical dimension
    Id = I(loc_dim)

    D = 2 + sum([abs(j - i) for (i, j, _, _, _) in two_body]) # Total bond dimension

    W = zeros(type, D, D, loc_dim, loc_dim)

    @views W[1, 1, :, :] .= Id #Starting state
    @views W[D, D, :, :] .= Id

    for (i, O, c1) in one_body
        @views W[1, D, :, :] .+= c1 .* O  #local operator sector
    end

    next_chan = 2
    for (i, j, Oi, Oj, c2) in two_body
        @assert i < j "Two‐site terms must be ordered: i < j"

        d = j - i

        start = next_chan
        finish = next_chan + d - 1
        next_chan = finish + 1

        @views W[1, start, :, :] .+= Oi #Insertion of first operator

        for k in (i + 1):(j - 1)
            @views W[start + (k - (i + 1)), start + (k - (i + 1)) + 1, :, :] .+= Id #Propagation of identities through unaffected sites
        end
        @views W[finish, D, :, :] .+= c2 .* Oj #Insertion of second operator
    end

    W_1 = W[:, end, :, :] #Vector for the first and last site
    W_L = W[1, :, :, :]

    return MPO([W_1, [W for _ in 2:(L - 1)]..., W_L])
end

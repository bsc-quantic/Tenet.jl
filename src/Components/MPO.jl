using DelegatorTraits
using Tangles
using Random

abstract type AbstractMPO <: AbstractTangle end

defaultorder(::Type{<:AbstractMPO}) = (:o, :i, :l, :r)

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

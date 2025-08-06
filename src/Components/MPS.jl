using Tangles
using Random

abstract type AbstractMPS <: AbstractMPO end

defaultorder(::Type{<:AbstractMPS}) = (:l, :r, :o)

"""
    MatrixProductState

A Matrix Product State Tensor Network.
"""
mutable struct MatrixProductState <: AbstractMPS
    const tn::GenericTensorNetwork
    form::CanonicalForm
end

MatrixProductState(tn::GenericTensorNetwork) = MatrixProductState(tn, MixedCanonical(sites(tn)))

const MPS = MatrixProductState

ImplementorTrait(interface, tn::MPS) = ImplementorTrait(interface, tn.tn)
function DelegatorTrait(interface, tn::MPS)
    if ImplementorTrait(interface, tn.tn) === Implements()
        DelegateToField{:tn}()
    else
        DontDelegate()
    end
end

Base.copy(tn::MPS) = MPS(copy(tn.tn), copy(tn.form))
Base.length(tn::MPS) = nsites(tn) # as required by Stefano but do not use, as it may be removed

CanonicalForm(tn::MPS) = tn.form
function unsafe_setform!(tn::MPS, form)
    @assert form isa NonCanonical || form isa MixedCanonical || form isa BondCanonical
    tn.form = form
    return tn
end

"""
    MPS(arrays::Vector{<:AbstractArray}; order=defaultorder(MPS))

Create a [`NonCanonical`](@ref) or [`MixedCanonical`](@ref) [`MPS`](@ref) from a vector of arrays.

# Keyword Arguments

  - `order` The order of the indices in the arrays. Defaults to `(:l, :r, :o)`.
"""
function MPS(arrays::AbstractVector{<:AbstractArray}; order=defaultorder(MPS)) # , check=true)
    @assert ndims(arrays[1]) == 2 "First array must have 2 dimensions"
    @assert all(==(3) ∘ ndims, arrays[2:(end - 1)]) "All arrays must have 3 dimensions"
    @assert ndims(arrays[end]) == 2 "Last array must have 2 dimensions"
    issetequal(order, defaultorder(MPS)) ||
        throw(ArgumentError("order must be a permutation of $(String.(defaultorder(MPS)))"))

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
        hasbond(tn, bond"$i-$isup") ||
            hasind(tn, Index(bond"$i-$isup")) && setbond!(tn, Index(bond"$i-$isup"), bond"$i-$isup")
        hasbond(tn, bond"$isub-$i") ||
            hasind(tn, Index(bond"$isub-$i")) && setbond!(tn, Index(bond"$isub-$i"), bond"$isub-$i")
    end

    return MPS(tn)
end

# TODO normalize as we canonize for numerical stability
# TODO different input/output physical dims
# TODO let choose the orthogonality center
# TODO add form information
"""
    Base.rand(rng::Random.AbstractRNG, ::Type{MPS}; n, maxdim, eltype=Float64, physdim=2)

Create a random [`MPS`](@ref) Tensor Network in the MixedCanonical form where all tensors are right-canonical (ortogonality
center at the first site). In order to avoid norm explosion issues, the tensors are orthogonalized by LQ factorization.

# Keyword Arguments

  - `n` The number of sites.
  - `maxdim` The maximum bond dimension.  If it is `nothing`, the maximum bond dimension increases exponentially with the number of sites up to `physdim^(n ÷ 2)`.
  - `eltype` The element type of the tensors. Defaults to `Float64`.
  - `physdim` The physical or output dimension of each site. Defaults to 2.
"""
function Base.rand(rng::Random.AbstractRNG, ::Type{MPS}; n, maxdim=128, eltype=Float64, physdim=2)
    if maxdim == 1
        return convert(MPS, rand(rng, ProductState; n, eltype, physdim))
    end

    p = physdim
    T = eltype
    χ = isnothing(maxdim) ? checked_pow(p, n ÷ 2) : maxdim

    arrays::Vector{AbstractArray{T,N} where {N}} = map(1:n) do i
        χl, χr = let after_mid = i > n ÷ 2, i = (n + 1 - abs(2i - n - 1)) ÷ 2
            χl = min(
                χ,
                try
                    checked_pow(p, i - 1)
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
                    checked_pow(p, i)
                catch e
                    if e isa OverflowError
                        typemax(Int)
                    else
                        rethrow(e)
                    end
                end,
            )

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

function Base.convert(::Type{MPS}, old_tn::ProductState)
    n = nsites(old_tn)
    tn = GenericTensorNetwork()
    for i in 1:n
        _tensor = tensor_at(old_tn, site"$i")

        _array = if i == 1 || i == n
            reshape(parent(_tensor), 1, length(_tensor))
        else
            reshape(parent(_tensor), 1, 1, length(_tensor))
        end

        _inds = if i == 1
            [Index(bond"$i-$(i + 1)"), Index(plug"$i")]
        elseif i == n
            [Index(bond"$(i - 1)-$i"), Index(plug"$i")]
        else
            [Index(bond"$(i - 1)-$i"), Index(bond"$i-$(i + 1)"), Index(plug"$i")]
        end

        new_tensor = Tensor(_array, _inds)

        addtensor!(tn, new_tensor)
        setsite!(tn, new_tensor, site"$i")
        setplug!(tn, Index(plug"$i"), plug"$i")
        if i != 1 && !hasbond(tn, bond"$(i - 1)-$i")
            setbond!(tn, Index(bond"$(i - 1)-$i"), bond"$(i - 1)-$i")
        end
        if i != n && !hasbond(tn, bond"$i-$(i + 1)")
            setbond!(tn, Index(bond"$i-$(i + 1)"), bond"$i-$(i + 1)")
        end
    end

    return MPS(tn, MixedCanonical(sites(tn)))
end

function bondsizes(psi::MPS; sorted::Bool=true)
    bs = bonds(psi)
    sorted_bonds = sorted ? sort(bs; by=b -> minmax(sites(b)...)) : bs
    return [size(psi, ind_at(psi, b)) for b in sorted_bonds]
end

maxbondsize(psi::MPS) = maximum(bondsizes(psi; sorted=false))

function Base.show(io::IO, psi::MPS)
    print(io, "MPS (#tensors=$(ntensors(psi)), #inds=$(ninds(psi)), maxbondsize=$(maxbondsize(psi)))")
end


""" Expectation value for a one-site operator `op` on site `op_site` of mps `psi`, brings to canonical form. 
This makes a copy of the MPS """
function expect_1site!(psi::MPS, op::AbstractArray, op_site::Int)
        canonize!(psi, MixedCanonical(site"$(op_site)"))
        plug_ind = ind_at(psi, plug"$(op_site)")
        temp_ind = Index(:_temp_bra)
        ev = binary_einsum(psi[op_site], Tensor(op, [temp_ind, plug_ind]))
        ev = binary_einsum(ev, conj(replace(psi[op_site], plug_ind => temp_ind)))

        return ev
end

function expect_1site(psi::MPS, op::AbstractArray, op_site::Int)
        psi = copy(psi)
        expect_1site!(psi, op, op_site)
end
      

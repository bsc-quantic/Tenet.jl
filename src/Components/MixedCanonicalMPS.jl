using DelegatorTraits
using Bijections
using ArgCheck

mutable struct MixedCanonicalMatrixProductState <: AbstractMPS
    const tensors::Vector{Tensor}
    orthog_center::MixedCanonical
    const plugs::Bijection{Plug,Index,Dict{Plug,Index},Dict{Index,Plug}}
    unsafe::Ref{Union{Nothing,TenetCore.UnsafeScope}}
end

const MixedCanonicalMPS = MixedCanonicalMatrixProductState

function Base.copy(tn::MixedCanonicalMPS)
    unsafe = Ref{Union{Nothing,TenetCore.UnsafeScope}}(tn.unsafe[])
    new_tn = MixedCanonicalMPS(copy(tn.tensors), copy(tn.orthog_center), copy(tn.plugs), unsafe)

    # register the new copy to the proper UnsafeScope
    !isnothing(unsafe[]) && push!(unsafe[].refs, WeakRef(new_tn))

    return new_tn
end

function Base.zero(tn::MixedCanonicalMPS)
    tn = copy(tn)

    for tensor in tensors(tn)
        replace_tensor!(tn, tensor, zero(tensor))
    end

    return tn
end

function MixedCanonicalMPS(arrays; form=MixedCanonical(CartesianSite.(1:length(arrays))), kwargs...)
    MixedCanonicalMPS(form, arrays; kwargs...)
end

function MixedCanonicalMPS(_form::MixedCanonical, arrays; order=defaultorder(MixedCanonicalMPS), unsafe=nothing) # , check=true)
    @assert ndims(arrays[1]) == 2 "First array must have 2 dimensions"
    @assert all(==(3) ∘ ndims, arrays[2:(end - 1)]) "All arrays must have 3 dimensions"
    @assert ndims(arrays[end]) == 2 "Last array must have 2 dimensions"
    issetequal(order, defaultorder(MixedCanonicalMPS)) ||
        throw(ArgumentError("order must be a permutation of $(String.(defaultorder(MixedCanonicalMPS)))"))

    _tensors = Tensor[]
    _plugs = Bijection{Plug,Index}()

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

        _inds = map(local_order) do dir
            if dir == :o
                Index(plug"i")
            elseif dir == :r
                Index(bond"i-isup")
            elseif dir == :l
                Index(bond"isub-i")
            else
                throw(ArgumentError("Invalid direction: $dir"))
            end
        end |> collect

        _tensor = Tensor(array, _inds)
        push!(_tensors, _tensor)
        _plugs[plug"i"] = Index(plug"i")
    end

    return MixedCanonicalMPS(_tensors, _form, _plugs, Ref{Union{Nothing,TenetCore.UnsafeScope}}(unsafe))
end

# UnsafeScopeable implementation
ImplementorTrait(::TenetCore.UnsafeScopeable, ::MixedCanonicalMPS) = Implements()

TenetCore.get_unsafe_scope(tn::MixedCanonicalMPS) = tn.unsafe[]
TenetCore.set_unsafe_scope!(tn::MixedCanonicalMPS, uc) = tn.unsafe[] = uc

# TensorNetwork interface
ImplementorTrait(::TensorNetwork, ::MixedCanonicalMPS) = Implements()

TenetCore.all_tensors(tn::MixedCanonicalMPS) = collect(tn.tensors)
TenetCore.all_tensors_iter(tn::MixedCanonicalMPS) = tn.tensors

TenetCore.tensor_at(tn::MixedCanonicalMPS, site::CartesianSite{1}) = tn.tensors[site.id[1]]
TenetCore.ind_at(tn::MixedCanonicalMPS, plug::Plug) = tn.plugs[plug]

function TenetCore.ind_at(tn::MixedCanonicalMPS, bond::Bond)
    @argcheck hasbond(tn, bond) "Bond $bond not found"
    inds(tensor_at(tn, sites(bond)[1])) ∩ inds(tensor_at(tn, sites(bond)[2])) |> only
end

TenetCore.addtensor!(::MixedCanonicalMPS, args...) = error("MixedCanonicalMPS doesn't allow `addtensor!`")
TenetCore.rmtensor!(::MixedCanonicalMPS, args...) = error("MixedCanonicalMPS doesn't allow `rmtensor!`")

function TenetCore.replace_tensor!(tn::MixedCanonicalMPS, old, new)
    old === new && return tn
    @argcheck issetequal(inds(new), inds(old)) "New tensor must have the same indices as the old tensor"
    for ind in inds(old)
        if !TenetCore.isscoped(tn) && size(new, ind) != size(old, ind)
            throw(DimensionMismatch("New tensor must have the same size as the old tensor for index $ind"))
        end
    end

    i = findfirst(Base.Fix1(===, old), all_tensors(tn))
    if isnothing(i)
        throw(ArgumentError("Tensor not found in MixedCanonicalMPS"))
    end

    tn.tensors[i] = new
    return tn
end

function TenetCore.replace_ind!(tn::MixedCanonicalMPS, old, new)
    # replace tensors
    for (i, tensor) in enumerate(tn.tensors)
        if old ∈ inds(tensor)
            tn.tensors[i] = replace(tensor, old => new)
        end
    end

    # update plugs
    if hasvalue(tn.plugs, old)
        _plug = inv(tn.plugs)[old]
        tn.plugs[_plug] = new
    end

    return tn
end

function TenetCore.replace_ind!(tn::MixedCanonicalMPS, old_new::AbstractDict)
    # replace tensors
    for (i, tensor) in enumerate(tn.tensors)
        if !isdisjoint(inds(tensor), keys(old_new))
            tn.tensors[i] = Tensor(parent(tensor), [get(old_new, ind, ind) for ind in inds(tensor)])
        end
    end

    # update plugs
    for (old_ind, new_ind) in old_new
        if hasvalue(tn.plugs, old_ind)
            tn.plugs[tn.plugs(old_ind)] = new_ind
        end
    end

    return tn
end

# Lattice interface
ImplementorTrait(::TenetCore.Lattice, ::MixedCanonicalMPS) = Implements()

TenetCore.all_sites(tn::MixedCanonicalMPS) = CartesianSite.(1:length(tn.tensors))
TenetCore.all_bonds(tn::MixedCanonicalMPS) = [Bond(CartesianSite.((i, i + 1))...) for i in 1:(length(tn.tensors) - 1)]

function TenetCore.site_at(tn::MixedCanonicalMPS, tensor::Tensor)
    i = findfirst(all_tensors_iter(tn)) do t
        t === tensor
    end
    isnothing(i) && throw(ArgumentError("Tensor not found"))
    return site"i"
end

function TenetCore.bond_at(tn::MixedCanonicalMPS, ind::Index)
    _tensors = tensors_with_inds(tn, ind)
    length(_tensors) != 2 && throw(ArgumentError("Bond must be between two tensors"))
    _sites = site_at.(Ref(tn), _tensors)
    return Bond(_sites...)
end

TenetCore.setsite!(::MixedCanonicalMPS, args...) = error("MixedCanonicalMPS doesn't allow `setsite!`")
TenetCore.setbond!(::MixedCanonicalMPS, args...) = error("MixedCanonicalMPS doesn't allow `setbond!`")
TenetCore.unsetsite!(::MixedCanonicalMPS, site) = error("MixedCanonicalMPS doesn't allow `unsetsite!`")
TenetCore.unsetbond!(::MixedCanonicalMPS, bond) = error("MixedCanonicalMPS doesn't allow `unsetbond!`")

# Pluggable interface
ImplementorTrait(::TenetCore.Pluggable, ::MixedCanonicalMPS) = Implements()

TenetCore.all_plugs(tn::MixedCanonicalMPS) = collect(keys(tn.plugs))
TenetCore.all_plugs_iter(tn::MixedCanonicalMPS) = keys(tn.plugs)
TenetCore.hasplug(tn::MixedCanonicalMPS, plug) = haskey(tn.plugs, plug)
TenetCore.nplugs(tn::MixedCanonicalMPS) = length(tn.plugs)

TenetCore.plug_at(tn::MixedCanonicalMPS, ind::Index) = tn.plugs(ind)

TenetCore.setplug!(::MixedCanonicalMPS, args...) = error("MixedCanonicalMPS doesn't allow `setplug!`")
TenetCore.unsetplug!(::MixedCanonicalMPS, args...) = error("MixedCanonicalMPS doesn't allow `unsetplug!`")

# CanonicalForm trait
CanonicalForm(tn::MixedCanonicalMPS) = tn.orthog_center

# TODO normalize as we canonize for numerical stability
# TODO different input/output physical dims
# TODO let choose the orthogonality center
# TODO add form information
"""
    Base.rand(rng::Random.AbstractRNG, ::Type{MixedCanonicalMPS}; n, maxdim, eltype=Float64, physdim=2)

Create a random [`MPS`](@ref) Tensor Network in the MixedCanonical form where all tensors are right-canonical (ortogonality
center at the first site). In order to avoid norm explosion issues, the tensors are orthogonalized by LQ factorization.

# Keyword Arguments

  - `n` The number of sites.
  - `maxdim` The maximum bond dimension.  If it is `nothing`, the maximum bond dimension increases exponentially with the number of sites up to `physdim^(n ÷ 2)`.
  - `eltype` The element type of the tensors. Defaults to `Float64`.
  - `physdim` The physical or output dimension of each site. Defaults to 2.
"""
function Base.rand(rng::Random.AbstractRNG, ::Type{MixedCanonicalMPS}; n, maxdim=nothing, eltype=Float64, physdim=2)
    p = physdim
    T = eltype
    χ = isnothing(maxdim) ? p^(n ÷ 2) : maxdim

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

    return MixedCanonicalMPS(arrays; order=(:l, :o, :r))
end

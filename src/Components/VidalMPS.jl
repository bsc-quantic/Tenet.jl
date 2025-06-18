using DelegatorTraits
using QuantumTags
using LinearAlgebra
using ArgCheck
using Bijections
using Tangles

struct LambdaSite{B} <: Site
    bond::B
end

# required for set-like equivalence of `Bond` to work on dictionaries
Base.:(==)(s1::LambdaSite, s2::LambdaSite) = s1.bond == s2.bond
Base.hash(s::LambdaSite, h::UInt) = hash(s.bond, h)

QuantumTags.bond(s::LambdaSite) = s.bond
QuantumTags.sites(s::LambdaSite) = sites(bond(s))

struct VidalMatrixProductState <: AbstractMPS
    Γ::Vector{Tensor}
    Λ::Vector{Tensor}
    plugs::Bijection{Plug,Index,Dict{Plug,Index},Dict{Index,Plug}}
end

const VidalMPS = VidalMatrixProductState

Base.copy(tn::VidalMPS) = VidalMPS(copy(tn.Γ), copy(tn.Λ), copy(tn.plugs))

function VidalMPS(Γ::AbstractVector{<:AbstractArray}, Λ::AbstractVector{<:AbstractArray}; order=defaultorder(VidalMPS))
    @assert ndims(Γ[1]) == 2 "First Γ array must have 2 dimensions"
    @assert all(==(3) ∘ ndims, Γ[2:(end - 1)]) "Inner Γ arrays must have 3 dimensions"
    @assert ndims(Γ[end]) == 2 "Last Γ array must have 2 dimensions"
    @assert all(==(1) ∘ ndims, Λ) "Λ arrays must have 1 dimension"
    if !issetequal(order, defaultorder(VidalMPS))
        throw(ArgumentError("order must be a permutation of $(String.(defaultorder(VidalMPS)))"))
    end

    _plugs = Bijection{Plug,Index}()

    Λ = map(enumerate(Λ)) do (i, λ)
        a = CartesianSite(i)
        b = CartesianSite(i + 1)
        _site = LambdaSite(bond"a-b")
        return Tensor(Diagonal(λ), Index.([Bond(a, _site), Bond(_site, b)]))
    end

    Γ = map(enumerate(Γ)) do (i, g)
        isub = LambdaSite(Bond(i - 1, i))
        isup = LambdaSite(Bond(i, i + 1))

        local_order = if i == 1
            filter(x -> x != :l, order)
        elseif i == length(Γ)
            filter(x -> x != :r, order)
        else
            order
        end

        inds = map(local_order) do dir
            if dir == :o
                Index(plug"i")
            elseif dir == :r
                Index(Bond(i, isup))
            elseif dir == :l
                Index(Bond(isub, i))
            else
                throw(ArgumentError("Invalid direction: $dir"))
            end
        end |> collect

        _plugs[plug"i"] = Index(plug"i")
        return Tensor(g, inds)
    end

    return VidalMPS(Γ, Λ, _plugs)
end

# TensorNetwork interface
ImplementorTrait(::Tangles.TensorNetwork, ::VidalMPS) = Implements()

Tangles.all_tensors(tn::VidalMPS) = [tn.Γ; tn.Λ]
Tangles.all_tensors_iter(tn::VidalMPS) = Iterators.flatten((tn.Γ, tn.Λ))

Tangles.tensor_at(tn::VidalMPS, site::CartesianSite{1}) = tn.Γ[site.id[1]]
function Tangles.tensor_at(tn::VidalMPS, s::LambdaSite{Bond{CartesianSite{1},CartesianSite{1}}})
    a, b = sites(s)
    # TODO do this check better
    i = a.id[1]
    @assert i == b.id[1] - 1 "Lambda sites must be consecutive"
    return tn.Λ[i]
end

Tangles.ind_at(tn::VidalMPS, p::Plug) = tn.plugs[p]

Tangles.addtensor!(tn::VidalMPS, args...) = error("VidalMPS doesn't allow `addtensor!`")
Tangles.rmtensor!(tn::VidalMPS, args...) = error("VidalMPS doesn't allow `rmtensor!`")

function Tangles.replace_tensor!(tn::VidalMPS, old, new)
    old === new && return tn

    i = findfirst(Base.Fix1(===, old), tn.Γ)
    if !isnothing(i)
        @argcheck issetequal(inds(new), inds(old)) "New tensor must have the same indices as the old tensor"
        tn.Γ[i] = new
        return tn
    end

    i = findfirst(Base.Fix1(===, old), tn.Λ)
    if !isnothing(i)
        @argcheck issetequal(inds(new), inds(old)) "New tensor must have the same indices as the old tensor"
        @argcheck isdiag(parent(new)) "New tensor must be diagonal for VidalMPS"
        tn.Λ[i] = new
        return tn
    end

    throw(ArgumentError("Tensor not found in VidalMPS"))
end

function Tangles.replace_ind!(tn::VidalMPS, old, new)
    # replace tensors
    for (i, tensor) in enumerate(tn.Γ)
        if old ∈ inds(tensor)
            tn.Γ[i] = replace(tensor, old => new)
        end
    end

    for (i, tensor) in enumerate(tn.Λ)
        if old ∈ inds(tensor)
            tn.Λ[i] = replace(tensor, old => new)
        end
    end

    # update plugs
    if hasvalue(tn.plugs, old)
        _plug = inv(tn.plugs)[old]
        tn.plugs[_plug] = new
    end

    return tn
end

# Lattice interface
ImplementorTrait(::Tangles.Lattice, ::VidalMPS) = Implements()

function Tangles.all_sites(tn::VidalMPS)
    [
        CartesianSite.(1:length(tn.Γ))
        LambdaSite.(Bond.(CartesianSite.(1:(length(tn.Γ) - 1)), CartesianSite.(2:length(tn.Γ))))
    ]
end

function Tangles.all_bonds(tn::VidalMPS)
    _bonds = Bond[]
    for i in 1:(length(tn.Γ) - 1)
        real_bond = Bond(CartesianSite(i), CartesianSite(i + 1))
        lamba_site = LambdaSite(real_bond)
        push!(_bonds, Bond(CartesianSite(i), lamba_site))
        push!(_bonds, Bond(lamba_site, CartesianSite(i + 1)))
    end
    return _bonds
end

Tangles.site_at(tn::VidalMPS, tensor::Tensor) = begin
    i = findfirst(Base.Fix1(===, tensor), tn.Γ)
    if !isnothing(i)
        return site"i"
    end

    i = findfirst(Base.Fix1(===, tensor), tn.Λ)
    if !isnothing(i)
        j = i + 1
        return LambdaSite(bond"i-j")
    end

    throw(ArgumentError("Tensor not found in VidalMPS"))
end

function Tangles.bond_at(tn::VidalMPS, ind::Index)
    _tensors = tensors_with_inds(tn, ind)
    length(_tensors) != 2 || throw(ArgumentError("Bond must be between two tensors"))
    _sites = site_at.(Ref(tn), _tensors)
    return Bond(_sites...)
end

Tangles.setsite!(::VidalMPS, args...) = error("VidalMPS doesn't allow `setsite!`")
Tangles.setbond!(::VidalMPS, args...) = error("VidalMPS doesn't allow `setbond!`")
Tangles.unsetsite!(::VidalMPS, site) = error("VidalMPS doesn't allow `unsetsite!`")
Tangles.unsetbond!(::VidalMPS, bond) = error("VidalMPS doesn't allow `unsetbond!`")

# Pluggable interface
ImplementorTrait(::Tangles.Pluggable, ::VidalMPS) = Implements()

Tangles.all_plugs(tn::VidalMPS) = collect(keys(tn.plugs))
Tangles.all_plugs_iter(tn::VidalMPS) = keys(tn.plugs)
Tangles.hasplug(tn::VidalMPS, plug) = haskey(tn.plugs, plug)
Tangles.nplugs(tn::VidalMPS) = length(tn.plugs)

Tangles.plug_at(tn::VidalMPS, ind::Index) = tn.plugs(ind)

Tangles.setplug!(::VidalMPS, args...) = error("VidalMPS doesn't allow `setplug!`")
Tangles.unsetplug!(::VidalMPS, args...) = error("VidalMPS doesn't allow `unsetplug!`")

# CanonicalForm trait
CanonicalForm(::VidalMPS) = VidalGauge()

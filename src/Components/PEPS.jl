using DelegatorTraits
using QuantumTags
using Networks
using Tangles

struct ProjectedEntangledPairState <: AbstractTangle
    tn::GenericTensorNetwork
end

const PEPS = ProjectedEntangledPairState
Base.copy(tn::PEPS) = PEPS(copy(tn.tn))

defaultorder(::Type{PEPS}) = (:l, :r, :u, :d, :o)

function PEPS(arrays::AbstractMatrix{<:AbstractArray}; order=defaultorder(PEPS))
    @argcheck issetequal(order, defaultorder(PEPS)) "order must be a permutation of $(String.(defaultorder(PEPS)))"

    tn = GenericTensorNetwork()
    m, n = size(arrays)

    for I in eachindex(IndexCartesian(), arrays)
        (i, j) = Tuple(I)
        let dirs = collect(order)
            i == 1 && filter!(!=(:u), dirs)
            i == m && filter!(!=(:d), dirs)
            j == 1 && filter!(!=(:l), dirs)
            j == n && filter!(!=(:r), dirs)

            _inds = map(dirs) do dir
                if dir === :l
                    Index(Bond(CartesianSite(i, j), CartesianSite(i, j - 1)))
                elseif dir === :r
                    Index(Bond(CartesianSite(i, j), CartesianSite(i, j + 1)))
                elseif dir === :u
                    Index(Bond(CartesianSite(i, j), CartesianSite(i - 1, j)))
                elseif dir === :d
                    Index(Bond(CartesianSite(i, j), CartesianSite(i + 1, j)))
                elseif dir === :o
                    Index(plug"i,j")
                else
                    throw(ArgumentError("Invalid direction: $dir"))
                end
            end

            _tensor = Tensor(arrays[i, j], _inds)
            addtensor!(tn, _tensor)
            setsite!(tn, _tensor, CartesianSite(i, j))

            _bonds = map(filter(x -> x !== :o, dirs)) do dir
                if dir === :l
                    Bond(CartesianSite(i, j), CartesianSite(i, j - 1))
                elseif dir === :r
                    Bond(CartesianSite(i, j), CartesianSite(i, j + 1))
                elseif dir === :u
                    Bond(CartesianSite(i, j), CartesianSite(i - 1, j))
                elseif dir === :d
                    Bond(CartesianSite(i, j), CartesianSite(i + 1, j))
                end
            end

            for _bond in _bonds
                if !hasbond(tn, _bond)
                    setbond!(tn, Index(_bond), _bond)
                end
            end

            Tangles.setplug!(tn, Index(plug"i,j"), plug"i,j")
        end
    end

    return PEPS(tn)
end

# Network interface
DelegatorTraits.DelegatorTrait(::Network, ::PEPS) = DelegateToField{:tn}()

# Taggable interface
DelegatorTraits.DelegatorTrait(::Networks.Taggable, ::PEPS) = DelegateToField{:tn}()

Networks.tag_vertex!(::PEPS, args...) = error("PEPS doesn't allow `tag_vertex!`")
Networks.untag_vertex!(::PEPS, args...) = error("PEPS doesn't allow `untag_vertex!`")
Networks.tag_edge!(::PEPS, args...) = error("PEPS doesn't allow `tag_edge!`")
Networks.untag_edge!(::PEPS, args...) = error("PEPS doesn't allow `untag_edge!`")
Networks.replace_vertex_tag!(::PEPS, args...) = error("PEPS doesn't allow `replace_vertex_tag!`")
Networks.replace_edge_tag!(::PEPS, args...) = error("PEPS doesn't allow `replace_edge_tag!`")

# UnsafeScopeable interface
DelegatorTraits.DelegatorTrait(::Tangles.UnsafeScopeable, ::PEPS) = DelegateToField{:tn}()

# TensorNetwork interface
DelegatorTraits.DelegatorTrait(::TensorNetwork, ::PEPS) = DelegateToField{:tn}()

Tangles.addtensor!(::PEPS, args...) = error("PEPS doesn't allow `addtensor!`")
Tangles.rmtensor!(::PEPS, args...) = error("PEPS doesn't allow `rmtensor!`")

# Lattice interface
DelegatorTraits.DelegatorTrait(::Tangles.Lattice, ::PEPS) = DelegateToField{:tn}()

Tangles.setsite!(::PEPS, args...) = error("PEPS doesn't allow `setsite!`")
Tangles.setbond!(::PEPS, args...) = error("PEPS doesn't allow `setbond!`")
Tangles.unsetsite!(::PEPS, args...) = error("PEPS doesn't allow `unsetsite!`")
Tangles.unsetbond!(::PEPS, args...) = error("PEPS doesn't allow `unsetbond!`")

# Pluggable interface
DelegatorTraits.DelegatorTrait(::Tangles.Pluggable, ::PEPS) = DelegateToField{:tn}()

Tangles.setplug!(::PEPS, args...) = error("PEPS doesn't allow `setplug!`")
Tangles.unsetplug!(::PEPS, args...) = error("PEPS doesn't allow `unsetplug!`")

using DelegatorTraits
using QuantumTags
using Networks
using Tangles

struct ProjectedEntangledPairOperator <: AbstractTangle
    tn::GenericTensorNetwork
end

const PEPO = ProjectedEntangledPairOperator
Base.copy(tn::PEPO) = PEPO(copy(tn.tn))

defaultorder(::Type{PEPO}) = (:l, :r, :u, :d, :o, :i)

function PEPO(arrays::AbstractMatrix{<:AbstractArray}; order=defaultorder(PEPO))
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
                    Index(bond"$(i, j) - $(i, j - 1)")
                elseif dir === :r
                    Index(bond"$(i, j) - $(i, j + 1)")
                elseif dir === :u
                    Index(bond"$(i, j) - $(i - 1, j)")
                elseif dir === :d
                    Index(bond"$(i, j) - $(i + 1, j)")
                elseif dir === :o
                    Index(plug"$i, $j")
                elseif dir === :i
                    Index(plug"$i, $j'")
                else
                    throw(ArgumentError("Invalid direction: $dir"))
                end
            end

            _tensor = Tensor(arrays[i, j], _inds)
            addtensor!(tn, _tensor)
            setsite!(tn, _tensor, site"$i,$j")

            _bonds = map(filter(x -> x !== :o, dirs)) do dir
                if dir === :l
                    bond"$(i, j) - $(i, j - 1)"
                elseif dir === :r
                    bond"$(i, j) - $(i, j + 1)"
                elseif dir === :u
                    bond"$(i, j) - $(i - 1, j)"
                elseif dir === :d
                    bond"$(i, j) - $(i + 1, j)"
                end
            end

            for _bond in _bonds
                if !hasbond(tn, _bond)
                    setbond!(tn, Index(_bond), _bond)
                end
            end

            Tangles.setplug!(tn, Index(plug"$i, $j"), plug"$i, $j")
            Tangles.setplug!(tn, Index(plug"$i, $j'"), plug"$i, $j'")
        end
    end

    return PEPO(tn)
end

# Network interface
DelegatorTraits.DelegatorTrait(::Network, ::PEPO) = DelegateToField{:tn}()

# Taggable interface
DelegatorTraits.DelegatorTrait(::Taggable, ::PEPO) = DelegateToField{:tn}()

Networks.tag_vertex!(::PEPO, args...) = error("PEPO doesn't allow `tag_vertex!`")
Networks.untag_vertex!(::PEPO, args...) = error("PEPO doesn't allow `untag_vertex!`")
Networks.tag_edge!(::PEPO, args...) = error("PEPO doesn't allow `tag_edge!`")
Networks.untag_edge!(::PEPO, args...) = error("PEPO doesn't allow `untag_edge!`")
Networks.replace_vertex_tag!(::PEPO, args...) = error("PEPO doesn't allow `replace_vertex_tag!`")
Networks.replace_edge_tag!(::PEPO, args...) = error("PEPO doesn't allow `replace_edge_tag!`")

# UnsafeScopeable interface
DelegatorTraits.DelegatorTrait(::UnsafeScopeable, ::PEPO) = DelegateToField{:tn}()

# TensorNetwork interface
DelegatorTraits.DelegatorTrait(::TensorNetwork, ::PEPO) = DelegateToField{:tn}()

Tangles.addtensor!(::PEPO, args...) = error("PEPO doesn't allow `addtensor!`")
Tangles.rmtensor!(::PEPO, args...) = error("PEPO doesn't allow `rmtensor!`")

# Lattice interface
DelegatorTraits.DelegatorTrait(::Lattice, ::PEPO) = DelegateToField{:tn}()

Tangles.setsite!(::PEPO, args...) = error("PEPO doesn't allow `setsite!`")
Tangles.setbond!(::PEPO, args...) = error("PEPO doesn't allow `setbond!`")
Tangles.unsetsite!(::PEPO, args...) = error("PEPO doesn't allow `unsetsite!`")
Tangles.unsetbond!(::PEPO, args...) = error("PEPO doesn't allow `unsetbond!`")

# Pluggable interface
DelegatorTraits.DelegatorTrait(::Pluggable, ::PEPO) = DelegateToField{:tn}()

Tangles.setplug!(::PEPO, args...) = error("PEPO doesn't allow `setplug!`")
Tangles.unsetplug!(::PEPO, args...) = error("PEPO doesn't allow `unsetplug!`")

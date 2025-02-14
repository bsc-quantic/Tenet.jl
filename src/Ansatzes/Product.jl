using LinearAlgebra

abstract type AbstractProduct <: AbstractTensorNetwork end

"""
    Product <: AbstractTensorNetwork

A Tensor Network representing a product state.
"""
struct ProductState <: AbstractProduct
    tn::TensorNetwork
    pluggable::PluggableMixin
    ansatz::AnsatzMixin
end

"""
    Product <: AbstractTensorNetwork

A Tensor Network representing a product state.
"""
struct ProductOperator <: AbstractProduct
    tn::TensorNetwork
    pluggable::PluggableMixin
    ansatz::AnsatzMixin
end

# Tensor Network interface
Wraps(::Type{TensorNetwork}, ::AbstractProduct) = Yes()
TensorNetwork(tn::AbstractProduct) = tn.tn

# TODO copy, similar, zero

# PluggableMixin
Wraps(::Type{PluggableMixin}, ::AbstractProduct) = Yes()
PluggableMixin(tn::AbstractProduct) = tn.pluggable

# Ansatz interface
Wraps(::Type{AnsatzMixin}, ::AbstractProduct) = Yes()
AnsatzMixin(tn::AbstractProduct) = tn.ansatz

# effect handlers
handle!(tn::AbstractProduct, effect::ReplaceEffect{Pair{Symbol,Symbol}}) = handle!(tn.pluggable, effect)
handle!(tn::AbstractProduct, effect::ReplaceEffect{Pair{Tensor,Tensor}}) = handle!(tn.ansatz, effect)

# constructors
function ProductState(arrays::AbstractArray{<:AbstractVector})
    gen = IndexCounter()
    sitemap = Dict{Site,Symbol}([Site(i) => nextindex!(gen) for i in eachindex(IndexCartesian(), arrays)])
    bondmap = Dict{Bond,Symbol}()
    lanemap = Dict{Lane,Tensor}(
        map(eachindex(IndexCartesian(), arrays)) do i
            Lane(i) => Tensor(arrays[i], [sitemap[Site(i)]])
        end,
    )

    tn = TensorNetwork(values(lanemap))
    pluggable = PluggableMixin(sitemap)
    ansatz = AnsatzMixin(lanemap, bondmap)

    return ProductState(tn, pluggable, ansatz)
end

function ProductState(bitstr::String)
    arrays = map(bitstr) do bitchar
        if bitchar === '0'
            [1.0, 0.0]
        elseif bitchar === '1'
            [0.0, 1.0]
        elseif bitchar === '+'
            [1.0, 1.0] / sqrt(2)
        elseif bitchar === '-'
            [1.0, -1.0] / sqrt(2)
        elseif bitchar === '↑' || bitchar === 'u'
            [1.0, -1.0im] / sqrt(2)
        elseif bitchar === '↓' || bitchar === 'd'
            [-1.0, -1.0im] / sqrt(2)
        else
            throw(ArgumentError("invalid character: $bitchar"))
        end
    end

    return ProductState(arrays)
end

function ProductOperator(arrays::AbstractArray{<:AbstractMatrix})
    gen = IndexCounter()
    sitemap = Dict{Site,Symbol}(Site(i) => nextindex!(gen) for i in eachindex(IndexCartesian(), arrays))
    append!(sitemap, [Site(i; dual=true) => nextindex!(gen) for i in eachindex(IndexCartesian(), arrays)])
    bondmap = Dict{Bond,Symbol}()
    lanemap = Dict{Lane,Tensor}(
        map(eachindex(IndexCartesian(), arrays)) do i
            Lane(i) => Tensor(arrays[i], [sitemap[Site(i)], sitemap[Site(i; dual=true)]])
        end,
    )

    tn = TensorNetwork(values(lanemap))
    pluggable = PluggableMixin(sitemap)
    ansatz = AnsatzMixin(lanemap, bondmap)

    return ProductOperator(tn, pluggable, ansatz)
end

# derived methods
function LinearAlgebra.norm(tn::AbstractProduct, p::Real=2)
    mapreduce(*, tensors(tn)) do tensor
        norm(parent(tensor), p) # TODO is this implemented?
    end^(1//p)
end

function LinearAlgebra.opnorm(tn::ProductOperator; p::Real=2)
    return mapreduce(*, tensors(tn)) do tensor
        opnorm(parent(tensor), p)
    end^(1//p)
end

function LinearAlgebra.normalize!(tn::AbstractProduct; p::Real=2)
    for tensor in tensors(tn)
        normalize!(tensor, p)
    end
    return tn
end

function overlap(a::ProductState, b::ProductState)
    issetequal(lanes(a), lanes(b)) || throw(ArgumentError("Both `ProductStates` must have the same lanes"))
    return mapreduce(*, lanes(a)) do lane
        dot(tensor(a; at=lane), conj(tensor(b; at=lane)))
    end
end

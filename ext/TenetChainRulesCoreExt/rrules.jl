# `Tensor` constructor
Tensor_pullback(Δ) = (NoTangent(), Δ.data, NoTangent())
Tensor_pullback(Δ::AbstractArray) = (NoTangent(), Δ, NoTangent())
Tensor_pullback(Δ::AbstractThunk) = Tensor_pullback(unthunk(Δ))
ChainRulesCore.rrule(T::Type{<:Tensor}, data, inds) = T(data, inds), Tensor_pullback

# `TensorNetwork` constructor
TensorNetwork_pullback(Δ::TensorNetworkTangent) = (NoTangent(), tensors(Δ))
TensorNetwork_pullback(Δ::AbstractThunk) = TensorNetwork_pullback(unthunk(Δ))
ChainRulesCore.rrule(::Type{TensorNetwork}, tensors) = TensorNetwork(tensors), TensorNetwork_pullback

# `Base.conj` methods
conj_pullback(Δ::Tensor) = (NoTangent(), conj(Δ))
conj_pullback(Δ::Tangent{Tensor}) = (NoTangent(), conj(Δ))
conj_pullback(Δ::TensorNetworkTangent) = (NoTangent(), conj(Δ))
conj_pullback(Δ::AbstractThunk) = conj_pullback(unthunk(Δ))

ChainRulesCore.rrule(::typeof(Base.conj), tn::Tensor) = conj(tn), conj_pullback
ChainRulesCore.rrule(::typeof(Base.conj), tn::TensorNetwork) = conj(tn), conj_pullback

# `Base.getindex` methods
function ChainRulesCore.rrule(::typeof(Base.getindex), x::TensorNetwork, is::Symbol...; mul::Int=1)
    y = getindex(x, is...; mul)
    nots = map(_ -> NoTangent(), is)

    getindex_pullback(z::AbstractZero) = (NoTangent(), z, nots...)
    function getindex_pullback(ȳ)
        ithunk = InplaceableThunk(
            dx -> ∇getindex!(dx, unthunk(ȳ), is...; mul), @thunk(∇getindex(x, unthunk(ȳ), is...; mul))
        )
        return (NoTangent(), ithunk, nots...)
    end

    return y, getindex_pullback
end

# TODO multiplicity
∇getindex(x::TensorNetwork, dy, is...; mul) = TensorNetworkTangent(Tensor[dy])

# TODO multiplicity
∇getindex!(x::TensorNetwork, dy, is...; mul) = push!(x, dy)

# `Base.merge` methods
function ChainRulesCore.rrule(::typeof(Base.merge), a::TensorNetwork, b::TensorNetwork)
    c = merge(a, b)

    function merge_pullback(c̄)
        c̄ = unthunk(c̄)
        ā = TensorNetworkTangent([c̄.tensors[inds(tensor)] for tensor in tensors(a)])
        b̄ = TensorNetworkTangent([c̄.tensors[inds(tensor)] for tensor in tensors(b)])
        return NoTangent(), ā, b̄
    end

    return c, merge_pullback
end

# `contract` methods
# function ChainRulesCore.rrule(::typeof(contract), x::Tensor; dims)
#     y = contract(x; dims)

#     function contract_pullback(ȳ)
#         return (NoTangent(), ...) # TODO
#     end
#     contract_pullback(ȳ::AbstractThunk) = contract_pullback(unthunk(ȳ))

#     return y, contract_pullback
# end

# TODO fix projectors: indices get permuted but projector doesn't know how to handle that
function ChainRulesCore.rrule(::typeof(contract), a::Tensor, b::Tensor; kwargs...)
    c = contract(a, b; kwargs...)
    # proj_a = ProjectTo(a)
    # proj_b = ProjectTo(b)

    function contract_pullback(c̄)
        ā = @thunk contract(c̄, b)
        b̄ = @thunk contract(a, c̄)
        return (NoTangent(), ā, b̄)
    end
    contract_pullback(c̄::AbstractThunk) = contract_pullback(unthunk(c̄))

    return c, contract_pullback
end

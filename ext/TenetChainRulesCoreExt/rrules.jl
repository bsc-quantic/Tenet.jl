# `Tensor` constructor
Tensor_pullback(Δ) = (NoTangent(), Δ.data, NoTangent())
Tensor_pullback(Δ::AbstractArray) = (NoTangent(), Δ, NoTangent())
Tensor_pullback(Δ::AbstractThunk) = Tensor_pullback(unthunk(Δ))
ChainRulesCore.rrule(T::Type{<:Tensor}, data, inds) = T(data, inds), Tensor_pullback

# `TensorNetwork` constructor
TensorNetwork_pullback(Δ::TensorNetworkTangent) = (NoTangent(), tensors(Δ))
TensorNetwork_pullback(Δ::AbstractThunk) = TensorNetwork_pullback(unthunk(Δ))
function ChainRulesCore.rrule(::Type{TensorNetwork}, tensors)
    TensorNetwork(tensors), TensorNetwork_pullback
end

# `Base.conj` methods
conj_pullback(Δ::Tensor) = (NoTangent(), conj(Δ))
conj_pullback(Δ::Tangent{Tensor}) = (NoTangent(), conj(Δ))
conj_pullback(Δ::TensorNetworkTangent) = (NoTangent(), conj(Δ))
conj_pullback(Δ::AbstractThunk) = conj_pullback(unthunk(Δ))

function ChainRulesCore.rrule(::typeof(Base.conj), tn::Tensor)
    conj(tn), conj_pullback
end

function ChainRulesCore.rrule(::typeof(Base.conj), tn::TensorNetwork)
    conj(tn), conj_pullback
end

# `Base.getindex` methods
function ChainRulesCore.rrule(::typeof(Base.getindex), x::TensorNetwork, is::Symbol...; mul::Int = 1)
    y = getindex(x, is...; mul)
    nots = map(_ -> NoTangent(), is)

    getindex_pullback(z::AbstractZero) = (NoTangent(), z, nots...)
    function getindex_pullback(ȳ)
        ithunk = InplaceableThunk(
            dx -> ∇getindex!(dx, unthunk(ȳ), is...; mul),
            @thunk(∇getindex(x, unthunk(ȳ), is...; mul))
        )
        (NoTangent(), ithunk, nots...)
    end

    y, getindex_pullback
end

# TODO multiplicity
∇getindex(x::TensorNetwork, dy, is...; mul) = TensorNetworkTangent(Tensor[dy])

# TODO multiplicity
∇getindex!(x::TensorNetwork, dy, is...; mul) = push!(x, dy)

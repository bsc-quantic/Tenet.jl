using ValSplit

"""
    AbstractTensorNetwork

Abstract type for `TensorNetwork`-derived types.
Its subtypes must implement conversion or extraction of the underlying `TensorNetwork` by overloading the `TensorNetwork` constructor.

# Implementors interface

Any implementor of the `AbstractTensorNetwork` interface (currently only [`TensorNetwork`](@ref)) must define the following methods:

  - `inds`
  - `tensors`
  - `size`
"""
abstract type AbstractTensorNetwork end

"""
    Wraps(::Type{T}, x)

Return `Yes()` if the `x` wraps a `T` and `No()` otherwise.
"""
function Wraps end

"""
    tensors(tn::AbstractTensorNetwork)

Return a list of the `Tensor`s in the [`AbstractTensorNetwork`](@ref).

# Implementation details

  - As the tensors of a [`AbstractTensorNetwork`](@ref) are stored as keys of the `.tensormap` dictionary and it uses `objectid` as hash, order is not stable so it sorts for repeated evaluations.
"""
function tensors end

arrays(tn::AbstractTensorNetwork) = parent.(tensors(tn))

function inds end

inds(tn::AbstractTensorNetwork; kwargs...) = inds(sort_nt(values(kwargs)), tn)
inds(::@NamedTuple{}, tn::AbstractTensorNetwork) = inds((; set=:all), tn)
inds(kwargs::@NamedTuple{set::Symbol}, tn::AbstractTensorNetwork) = inds_set(tn, kwargs.set)
@valsplit inds_set(tn, Val(set::Symbol)) = error("Invalid set = $set")

# default implementations
inds_set(tn, ::Val{:all}) = inds_set(tn, Val(:all), Wraps(TensorNetwork, tn))
inds_set(tn, ::Val{:all}, ::Yes) = inds_set(TensorNetwork(tn), Val(:all))
inds_set(tn, ::Val{:all}, ::No) = mapreduce(inds, ∪, tensors(tn))

inds_set(tn, ::Val{:open}) = inds_set(tn, ::Val{:open}, Wraps(TensorNetwork, tn))
inds_set(tn, ::Val{:open}, ::Yes) = inds_set(TensorNetwork(tn), Val(::open))
function inds_set(tn, ::Val{:open}, ::No)
    histogram = hist(Iterators.flatten(Iterators.map(inds, tensors(tn))); init=Dict{Symbol,Int}())
    return last.(Iterators.filter(((k, c),) -> c == 1, histogram))
end

inds_set(tn, ::Val{:inner}) = inds_set(tn, ::Val{:inner}, Wraps(TensorNetwork, tn))
inds_set(tn, ::Val{:inner}, ::Yes) = inds_set(TensorNetwork(tn), Val(::inner))
function inds_set(tn, ::Val{:inner}, ::No)
    histogram = hist(Iterators.flatten(Iterators.map(inds, tensors(tn))); init=Dict{Symbol,Int}())
    return last.(Iterators.filter(((k, c),) -> c == 2, histogram))
end

inds_set(tn, ::Val{:hyper}) = inds_set(tn, ::Val{:hyper}, Wraps(TensorNetwork, tn))
inds_set(tn, ::Val{:hyper}, ::Yes) = inds_set(TensorNetwork(tn), Val(::hyper))
function inds_set(tn, ::Val{:hyper}, ::No)
    histogram = hist(Iterators.flatten(Iterators.map(inds, tensors(tn))); init=Dict{Symbol,Int}())
    return last.(Iterators.filter(((k, c),) -> c >= 3, histogram))
end

"""
    ninds(tn::TensorNetwork; kwargs...)

Return the number of indices in the `TensorNetwork`. It accepts the same keyword arguments as [`inds`](@ref).

See also: [`ntensors`](@ref)
"""
ninds(tn::AbstractTensorNetwork; kwargs...) = ninds(sort_nt(values(kwargs)), tn)

# dispatch due to performance reasons: see the implementation in src/TensorNetwork.jl
ninds(::@NamedTuple{}, tn::AbstractTensorNetwork) = ninds(::@NamedTuple{}, tn, Wraps(TensorNetwork, tn))
ninds(::@NamedTuple{}, tn::AbstractTensorNetwork, ::Yes) = ninds(::@NamedTuple{}, TensorNetwork(tn))
ninds(::@NamedTuple{}, tn::AbstractTensorNetwork, ::No) = ninds(::@NamedTuple{}, TensorNetwork(tn))
ninds(kwargs::NamedTuple, tn::AbstractTensorNetwork) = length(inds(kwargs, tn))

Base.size(tn::AbstractTensorNetwork) = size(tn, Wraps(TensorNetwork, tn))
Base.size(tn::AbstractTensorNetwork, ::Yes) = size(TensorNetwork(tn))
function Base.size(tn::AbstractTensorNetwork, ::No)
    sizes = Dict{Symbol,Int}()
    for tensor in tensors
        for ind in inds(tensor)
            sizes[ind] = get(sizes, ind, 0) + 1
        end
    end
    return sizes
end

Base.size(tn::AbstractTensorNetwork, i) = size(tn, i, Wraps(TensorNetwork, tn))
Base.size(tn::AbstractTensorNetwork, i, ::Yes) = size(TensorNetwork(tn), i)
function Base.size(tn::AbstractTensorNetwork, i, ::No)
    tensor = findfirst(t -> i ∈ inds(tensor), tensors(tn))
    isnothing(tensor) && throw(ArgumentError("Index $i not found in the Tensor Network"))
    return size(tensor, i)
end

Base.in(i::Symbol, tn::AbstractTensorNetwork) = inds(i, tn, Wraps(TensorNetwork, tn))
Base.in(i::Symbol, tn::AbstractTensorNetwork, ::Yes) = inds(i, TensorNetwork(tn))
Base.in(i::Symbol, tn::AbstractTensorNetwork, ::No) = i ∈ inds(tn)

Base.in(i::Tensor, tn::AbstractTensorNetwork) = in(i, tn, Wraps(TensorNetwork, tn))
Base.in(i::Tensor, tn::AbstractTensorNetwork, ::Yes) = in(i, TensorNetwork(tn))
Base.in(i::Tensor, tn::AbstractTensorNetwork, ::No) = i ∈ tensors(tn)

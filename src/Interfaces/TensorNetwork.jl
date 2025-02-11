using Base: AbstractVecOrTuple
using Graphs: Graphs
using EinExprs

"""
    AbstractTensorNetwork

Abstract type for `TensorNetwork`-derived types.
"""
abstract type AbstractTensorNetwork end

#=
Wraps-trait
=#

"""
    Wraps(::Type{T}, x)

Return `Yes()` if the `x` wraps a `T` and `No()` otherwise.
"""
function Wraps end

function unwrap end

#=
TensorNetwork interface
=#
"""
    tensors(tn; kwargs...)

Return a list of the [`Tensor`](@ref)s in the [`AbstractTensorNetwork`](@ref).
"""
function tensors end

tensors(tn::AbstractTensorNetwork; kwargs...) = tensors(sort_nt(values(kwargs)), tn)

tensors(::@NamedTuple{}, tn::AbstractTensorNetwork) = tensors((;), tn, Wraps(TensorNetwork, tn))
tensors(::@NamedTuple{}, tn::AbstractTensorNetwork, ::Yes) = tensors((;), TensorNetwork(tn))
function tensors(::@NamedTuple{}, tn::AbstractTensorNetwork, ::No)
    error(
        "$(typeof(tn)) must implement tensors(::@NamedTuple{}, $(typeof(tn))) or fulfill the `WrapsTensorNetwork` trait"
    )
end

tensors(kwargs::NamedTuple{(:contains,)}, tn::AbstractTensorNetwork) = tensors(tn; contains=[kwargs.contains]) # copy(TensorNetwork(tn).indexmap[contains])
function tensors(kwargs::@NamedTuple{contains::T}, tn::AbstractTensorNetwork) where {T<:AbstractVecOrTuple{Symbol}}
    return filter(t -> issubset(kwargs.contains, inds(t)), tensors(tn))
end

function tensors(kwargs::@NamedTuple{intersects::Symbol}, tn::AbstractTensorNetwork)
    tensors(tn; intersects=[kwargs.intersects])
end
function tensors(kwargs::@NamedTuple{intersects::T}, tn::AbstractTensorNetwork) where {T<:AbstractVecOrTuple{Symbol}}
    return filter(t -> !isdisjoint(inds(t), kwargs.intersects), tensors(tn))
end

"""
    ntensors(tn::AbstractTensorNetwork)

Return the number of tensors in the `TensorNetwork`. It accepts the same keyword arguments as [`tensors`](@ref).

See also: [`ninds`](@ref)
"""
ntensors(tn::AbstractTensorNetwork; kwargs...) = ntensors(values(kwargs), tn)

ntensors(kwargs::NamedTuple, tn::AbstractTensorNetwork) = length(tensors(kwargs, tn))

# dispatch due to performance reasons: see implementation in src/TensorNetwork.jl
ntensors(::@NamedTuple{}, tn::AbstractTensorNetwork) = ntensors((;), tn, Wraps(TensorNetwork, tn))
ntensors(::@NamedTuple{}, tn::AbstractTensorNetwork, ::Yes) = length(TensorNetwork(tn).tensormap)
ntensors(::@NamedTuple{}, tn::AbstractTensorNetwork, ::No) = length(tensors(tn))

"""
    inds(tn; kwargs...)

Return
"""
function inds end

inds(tn::AbstractTensorNetwork; kwargs...) = inds(sort_nt(values(kwargs)), tn)
inds(::@NamedTuple{}, tn::AbstractTensorNetwork) = inds((; set=:all), tn)

# default implementations
"""
    inds(tn; set = :all)

Return the names of the indices in the [`AbstractTensorNetwork`](@ref).

# Keyword Arguments

  - `set`

      + `:all` (default) All indices.
      + `:open` Indices only mentioned in one tensor.
      + `:inner` Indices mentioned at least twice.
      + `:hyper` Indices mentioned at least in three tensors.
"""
function inds(kwargs::@NamedTuple{set::Symbol}, tn::AbstractTensorNetwork)
    if kwargs.set === :all
        return mapreduce(inds, ∪, tensors(tn); init=Symbol[])
    elseif kwargs.set === :open
        histogram = hist(Iterators.flatten(Iterators.map(inds, tensors(tn))); init=Dict{Symbol,Int}())
        return first.(Iterators.filter(((k, c),) -> c == 1, histogram))
    elseif kwargs.set === :inner
        histogram = hist(Iterators.flatten(Iterators.map(inds, tensors(tn))); init=Dict{Symbol,Int}())
        return first.(Iterators.filter(((k, c),) -> c >= 2, histogram))
    elseif kwargs.set === :hyper
        histogram = hist(Iterators.flatten(Iterators.map(inds, tensors(tn))); init=Dict{Symbol,Int}())
        return first.(Iterators.filter(((k, c),) -> c >= 3, histogram))
    else
        error("Invalid set = $set")
    end
end

"""
    inds(tn; parallelto)

Return the indices parallel to an index in the [`AbstractTensorNetwork`](@ref).
"""
function inds(kwargs::NamedTuple{(:parallelto,)}, tn::AbstractTensorNetwork)
    candidates = filter!(!=(kwargs.parallelto), collect(mapreduce(inds, ∩, tensors(tn; contains=kwargs.parallelto))))
    return filter(candidates) do i
        length(tensors(tn; contains=i)) == length(tensors(tn; contains=kwargs.parallelto))
    end
end

"""
    ninds(tn::TensorNetwork; kwargs...)

Return the number of indices in the `TensorNetwork`. It accepts the same keyword arguments as [`inds`](@ref).

See also: [`ntensors`](@ref)
"""
ninds(tn::AbstractTensorNetwork; kwargs...) = ninds(sort_nt(values(kwargs)), tn)

# dispatch due to performance reasons: see implementation in src/TensorNetwork.jl
ninds(::@NamedTuple{}, tn::AbstractTensorNetwork) = ninds(@NamedTuple{}(), tn, Wraps(TensorNetwork, tn))
ninds(::@NamedTuple{}, tn::AbstractTensorNetwork, ::Yes) = ninds(@NamedTuple{}(), TensorNetwork(tn))
ninds(::@NamedTuple{}, tn::AbstractTensorNetwork, ::No) = ninds(@NamedTuple{}(), TensorNetwork(tn))
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

Base.in(i::Symbol, tn::AbstractTensorNetwork) = hasind(i, tn)
Base.in(i::Tensor, tn::AbstractTensorNetwork) = hastensor(i, tn)

hasind(i::Symbol, tn::AbstractTensorNetwork) = inds(i, tn, Wraps(TensorNetwork, tn))
hasind(i::Symbol, tn::AbstractTensorNetwork, ::Yes) = inds(i, TensorNetwork(tn))
hasind(i::Symbol, tn::AbstractTensorNetwork, ::No) = i ∈ inds(tn)

hastensor(i::Tensor, tn::AbstractTensorNetwork) = in(i, tn, Wraps(TensorNetwork, tn))
hastensor(i::Tensor, tn::AbstractTensorNetwork, ::Yes) = in(i, TensorNetwork(tn))
hastensor(i::Tensor, tn::AbstractTensorNetwork, ::No) = i ∈ tensors(tn)

"""
    replace!(tn::AbstractTensorNetwork, old => new...)
    replace(tn::AbstractTensorNetwork, old => new...)

Replace the element in `old` with the one in `new`. Depending on the types of `old` and `new`, the following behaviour is expected:

  - If `Symbol`s, it will correspond to a index renaming.
  - If `Tensor`s, first element that satisfies _egality_ (`≡` or `===`) will be replaced.
"""
@inline function Base.replace!(tn::T, old_new::P...) where {T<:AbstractTensorNetwork,P<:Pair}
    return invoke(replace!, Tuple{T,Base.AbstractVecOrTuple{P}}, tn, old_new)
end
@inline Base.replace!(tn::AbstractTensorNetwork, old_new::Dict) = replace!(tn, collect(old_new))

function Base.replace!(tn::AbstractTensorNetwork, old_new::Base.AbstractVecOrTuple{Pair})
    for pair in old_new
        replace!(tn, pair)
    end
    return tn
end

Base.replace!(tn::AbstractTensorNetwork) = tn
Base.replace(tn::AbstractTensorNetwork, old_new::Pair...) = replace(tn, old_new)
Base.replace(tn::AbstractTensorNetwork, old_new) = replace!(copy(tn), old_new)

function Base.replace!(tn::AbstractTensorNetwork, old_new::Base.AbstractVecOrTuple{Pair{Symbol,Symbol}})
    from, to = first.(old_new), last.(old_new)
    allinds = inds(tn)

    # condition: from ⊆ allinds
    from ⊆ allinds || throw(ArgumentError("set of old indices must be a subset of current indices"))

    # condition: from \ to ∩ allinds = ∅
    isdisjoint(setdiff(to, from), allinds) || throw(
        ArgumentError(
            "new indices must be either a element of the old indices or not an element of the TensorNetwork's indices",
        ),
    )

    overlap = from ∩ to
    if isempty(overlap)
        # no overlap so easy replacement
        for (f, t) in zip(from, to)
            replace!(tn, f => t)
        end
    else
        # overlap between old and new indices => need a temporary name `replace!`
        tmp = Dict([i => gensym(i) for i in from])

        # replace old indices with temporary names
        replace!(tn, tmp)

        # replace temporary names with new indices
        replace!(tn, [tmp[f] => t for (f, t) in zip(from, to)])
    end

    # return the final index mapping
    return tn
end

# generic methods on top of the interface
Base.eltype(tn::AbstractTensorNetwork) = promote_type(eltype.(tensors(tn))...)

"""
    arrays(tn::AbstractTensorNetwork; kwargs...)

Return a list of the arrays of in the `TensorNetwork`. It is equivalent to `parent.(tensors(tn; kwargs...))`.
"""
arrays(tn::AbstractTensorNetwork; kwargs...) = parent.(tensors(tn; kwargs...))

"""
    Base.collect(tn::AbstractTensorNetwork)

Return a list of the `Tensor`s in the `TensorNetwork`. It is equivalent to `tensors(tn)`.
"""
Base.collect(tn::AbstractTensorNetwork) = tensors(tn)

"""
    conj(tn::AbstractTensorNetwork)

Return a copy of the [`AbstractTensorNetwork`](@ref) with all tensors conjugated.
"""
function Base.conj(tn::AbstractTensorNetwork)
    tn = copy(tn)
    replace!(tn, tensors(tn) .=> conj.(tensors(tn)))
    return tn
end

function Base.conj!(tn::AbstractTensorNetwork)
    foreach(conj!, tensors(tn))
    return tn
end

function Graphs.neighbors(tn::AbstractTensorNetwork, tensor::Tensor; open::Bool=true)
    @assert tensor ∈ tn "Tensor not found in TensorNetwork"
    tensors = mapreduce(∪, inds(tensor)) do index
        Tenet.tensors(tn; intersects=index)
    end
    open && filter!(x -> x !== tensor, tensors)
    return tensors
end

function Graphs.neighbors(tn::AbstractTensorNetwork, i::Symbol; open::Bool=true)
    @assert i ∈ tn "Index $i not found in TensorNetwork"
    tensors = mapreduce(inds, ∪, Tenet.tensors(tn; intersects=i))
    # open && filter!(x -> x !== i, tensors)
    return tensors
end

"""
    einexpr(tn::AbstractTensorNetwork; optimizer = EinExprs.Greedy, output = inds(tn, :open), kwargs...)

Search a contraction path for the given [`AbstractTensorNetwork`](@ref) and return it as a `EinExpr`.

# Keyword Arguments

  - `optimizer` Contraction path optimizer. Check [`EinExprs`](https://github.com/bsc-quantic/EinExprs.jl) documentation for more info.
  - `outputs` Indices that won't be contracted. Defaults to open indices.
  - `kwargs` Options to be passed to the optimizer.

See also: [`contract`](@ref).
"""
function EinExprs.einexpr(tn::AbstractTensorNetwork; optimizer=Greedy, outputs=inds(tn; set=:open), kwargs...)
    return einexpr(
        optimizer,
        sum(
            [
                EinExpr(inds(tensor), Dict(index => size(tensor, index) for index in inds(tensor))) for
                tensor in tensors(tn)
            ];
            skip=outputs,
        );
        kwargs...,
    )
end

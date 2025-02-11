# This file defines the "TensorNetwork" interface

using Base: AbstractVecOrTuple
using Graphs: Graphs
using EinExprs

"""
    AbstractTensorNetwork

Abstract type for `TensorNetwork`-derived types.
"""
abstract type AbstractTensorNetwork end

"""
    Wraps(::Type{T}, x)

Return `Yes()` if the `x` wraps a `T` and `No()` otherwise.
"""
Base.@nospecializeinfer Wraps(@nospecialize(_::Type), @nospecialize(x)) = No()

# required methods
"""
    tensors(tn; kwargs...)

Return a list of the [`Tensor`](@ref)s in the Tensor Network.
"""
tensors(tn::AbstractTensorNetwork; kwargs...) = tensors(sort_nt(values(kwargs)), tn)

tensors(::@NamedTuple{}, tn::AbstractTensorNetwork) = tensors((;), tn, Wraps(TensorNetwork, tn))
tensors(::@NamedTuple{}, tn::AbstractTensorNetwork, ::Yes) = tensors((;), TensorNetwork(tn))
tensors(::@NamedTuple{}, tn::AbstractTensorNetwork, ::No) = throw(MethodError(tensors, ((;), tn)))

"""
    inds(tn; kwargs...)

Return the indices in the Tensor Network.

See also: [`tensors`](@ref)
"""
inds(tn::AbstractTensorNetwork; kwargs...) = inds(sort_nt(values(kwargs)), tn)
inds(::@NamedTuple{}, tn::AbstractTensorNetwork) = inds((; set=:all), tn)

"""
    copy(tn::AbstractTensorNetwork)

Return a _shallow_ copy of the Tensor Network; i.e. a new Tensor Network object with the **same** [`Tensor`](@ref)s. Chages to the copied Tensor Network won't affect the original one, but mutating data in them in one will affect both.

This method is used whenever a copy of the Tensor Network is needed, but want to save memory.
Keep in mind that [`Tensor`](@ref) is a immutable type, but it's data can be mutable.

If you want to mutate the data of a [`Tensor`](@ref) without affecting the original, use `replace!` as is much more memory efficient.
"""
Base.copy(tn::AbstractTensorNetwork)

# optional methods
"""
    hastensor(tn, tensor)

Return `true` if [`Tensor`](@ref) `tensor` is in the Tensor Network.

See also: [`hasind`](@ref)
"""
hastensor(tn::AbstractTensorNetwork, i::Tensor) = hastensor(tn, i, Wraps(TensorNetwork, tn))
hastensor(tn::AbstractTensorNetwork, i::Tensor, ::Yes) = hastensor(TensorNetwork(tn), i)
hastensor(tn::AbstractTensorNetwork, i::Tensor, ::No) = i ∈ tensors(tn)

"""
    hasind(tn, i)

Return `true` if index `i` is in the Tensor Network.

See also: [`hastensor`](@ref)
"""
hasind(tn::AbstractTensorNetwork, i::Symbol) = hasind(tn, i, Wraps(TensorNetwork, tn))
hasind(tn::AbstractTensorNetwork, i::Symbol, ::Yes) = hasind(TensorNetwork(tn), i)
hasind(tn::AbstractTensorNetwork, i::Symbol, ::No) = i ∈ inds(tn)

"""
    ntensors(tn::AbstractTensorNetwork)

Return the number of tensors in the `TensorNetwork`. It accepts the same keyword arguments as [`tensors`](@ref).

See also: [`ninds`](@ref)
"""
ntensors(tn::AbstractTensorNetwork; kwargs...) = ntensors(values(kwargs), tn)
ntensors(kwargs::NamedTuple, tn::AbstractTensorNetwork) = length(tensors(kwargs, tn))

# dispatch due to performance reasons: see implementation in src/TensorNetwork.jl
ntensors(::@NamedTuple{}, tn::AbstractTensorNetwork) = ntensors((;), tn, Wraps(TensorNetwork, tn))
ntensors(::@NamedTuple{}, tn::AbstractTensorNetwork, ::Yes) = ntensors(TensorNetwork(tn))
ntensors(::@NamedTuple{}, tn::AbstractTensorNetwork, ::No) = length(tensors(tn))

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

"""
    size(tn::AbstractTensorNetwork)

Return a dictionary with the indices as keys and their size as values.
"""
Base.size(tn::AbstractTensorNetwork) = size(tn, Wraps(TensorNetwork, tn))
Base.size(tn::AbstractTensorNetwork, ::Yes) = size(TensorNetwork(tn))
function Base.size(tn::AbstractTensorNetwork, ::No)
    sizes = Dict{Symbol,Int}()
    for tensor in tensors(tn)
        for ind in inds(tensor)
            sizes[ind] = get(sizes, ind, 0) + 1
        end
    end
    return sizes
end

"""
    size(tn::AbstractTensorNetwork, i)

Return the size of index `i` in the Tensor Network.
"""
Base.size(tn::AbstractTensorNetwork, i) = size(tn, i, Wraps(TensorNetwork, tn))
Base.size(tn::AbstractTensorNetwork, i, ::Yes) = size(TensorNetwork(tn), i)
function Base.size(tn::AbstractTensorNetwork, i, ::No)
    tensor = findfirst(t -> i ∈ inds(tensor), tensors(tn))
    isnothing(tensor) && throw(ArgumentError("Index $i not found in the Tensor Network"))
    return size(tensor, i)
end

# keyword methods
"""
    tensors(tn; contains)

Return the tensors containing **all** the given indices.
"""
tensors(kwargs::NamedTuple{(:contains,)}, tn::AbstractTensorNetwork) = tensors(kwargs, tn, Wraps(TensorNetwork, tn))
tensors(kwargs::NamedTuple{(:contains,)}, tn::AbstractTensorNetwork, ::Yes) = tensors(kwargs, TensorNetwork(tn))
function tensors(kwargs::NamedTuple{(:contains,)}, tn::AbstractTensorNetwork, ::No)
    tensors((; contains=kwargs.contains), tn, No())
end
function tensors(kwargs::@NamedTuple{contains::AbstractVecOrTuple{Symbol}}, tn::AbstractTensorNetwork, ::No)
    return filter(⊇(kwargs.contains) ∘ inds, tensors(tn))
end

# TODO dispatch to `TensorNetwork` and write optimized version for it in `src/TensorNetwork.jl`
"""
    tensors(tn; intersects)

Return the tensors intersecting with **at least one** of the given indices.
"""
function tensors(kwargs::@NamedTuple{intersects::T}, tn::AbstractTensorNetwork) where {T<:AbstractVecOrTuple{Symbol}}
    return filter(t -> !isdisjoint(inds(t), kwargs.intersects), tensors(tn))
end
function tensors(kwargs::@NamedTuple{intersects::Symbol}, tn::AbstractTensorNetwork)
    tensors(tn; intersects=[kwargs.intersects])
end

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
inds(kwargs::@NamedTuple{set::Symbol}, tn::AbstractTensorNetwork) = inds(kwargs, tn, Wraps(TensorNetwork, tn))
inds(kwargs::NamedTuple{(:set,)}, tn::AbstractTensorNetwork, ::Yes) = inds(kwargs, TensorNetwork(tn))
function inds(kwargs::@NamedTuple{set::Symbol}, tn::AbstractTensorNetwork, ::No)
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

# required mutating methods
"""
    push_inner!(tn::AbstractTensorNetwork, tensor)

Add a [`Tensor`](@ref) to the Tensor Network. This method is used by the [`push!`] method.
A user should not call this method directly.
"""
function push_inner! end

push_inner!(tn::AbstractTensorNetwork, tensor) = push_inner!(tn, tensor, Wraps(TensorNetwork, tn))
push_inner!(tn::AbstractTensorNetwork, tensor, ::Yes) = push_inner!(TensorNetwork(tn), tensor)
push_inner!(tn::AbstractTensorNetwork, tensor, ::No) = throw(MethodError(push_inner!, (tn, tensor)))

"""
    delete_inner!(tn::AbstractTensorNetwork, tensor)

Remove a [`Tensor`](@ref) from the Tensor Network. This method is used by the [`delete!`] method.
A user should not call this method directly.
"""
function delete_inner! end

delete_inner!(tn::AbstractTensorNetwork, tensor) = delete_inner!(tn, tensor, Wraps(TensorNetwork, tn))
delete_inner!(tn::AbstractTensorNetwork, tensor, ::Yes) = delete_inner!(TensorNetwork(tn), tensor)
delete_inner!(tn::AbstractTensorNetwork, tensor, ::No) = throw(MethodError(delete_inner!, (tn, tensor)))

"""
    contract_inner!(tn::AbstractTensorNetwork, ind)

Contract in-place the index `ind` in the Tensor Network. This method is used by the [`contract!`] method.
A user should not call this method directly.
"""
function contract_inner! end

contract_inner!(tn::AbstractTensorNetwork, tensor) = contract_inner!(tn, tensor, Wraps(TensorNetwork, tn))
contract_inner!(tn::AbstractTensorNetwork, tensor, ::Yes) = contract_inner!(TensorNetwork(tn), tensor)
contract_inner!(tn::AbstractTensorNetwork, tensor, ::No) = throw(MethodError(contract_inner!, (tn, tensor)))

# NOTE not really required to be in the interface, but needed to dispatch if `tn` wraps a `TensorNetwork`
tryprune!(tn::AbstractTensorNetwork, ind) = tryprune!(tn, ind, Wraps(TensorNetwork, tn))
tryprune!(tn::AbstractTensorNetwork, ind, ::Yes) = tryprune!(TensorNetwork(tn), ind)
tryprune!(::AbstractTensorNetwork, _, ::No) = nothing

# derived mutating methods
"""
    push!(tn::AbstractTensorNetwork, tensor)

Add a [`Tensor`](@ref) to the Tensor Network.
"""
function Base.push!(tn::AbstractTensorNetwork, t::Tensor)
    push_inner!(tn, t)
    handle(tn, PushEffect(t))
    return tn
end

"""
    append!(tn::AbstractTensorNetwork, tensors)
    append!(tn::AbstractTensorNetwork, other::AbstractTensorNetwork)

Add a tensors to a Tensor Network from a list of [`Tensor`](@ref)s or from another Tensor Network.

See also: [`push!`](@ref).
"""
Base.append!(tn::AbstractTensorNetwork, tensors) = (foreach(Base.Fix1(push!, tn), tensors); tn)
function Base.append!(tn::AbstractTensorNetwork, other::AbstractTensorNetwork)
    (foreach(Base.Fix1(push!, tn), tensors(other)); tn)
end

"""
    delete!(tn::AbstractTensorNetwork, tensor)

Remove a [`Tensor`](@ref) from the Tensor Network.

!!! warning

    [`Tensor`](@ref)s are identified in a [`TensorNetwork`](@ref) by their `objectid`, so you must pass the same object and not a copy.
"""
function Base.delete!(tn::AbstractTensorNetwork, t::Tensor)
    delete_inner!(tn, t)
    handle(tn, DeleteEffect(t))
    return tn
end

# TODO what do we do with this?
# """
#     delete!(tn::AbstractTensorNetwork, x)

# Like [`pop!`](@ref) but return the Tensor Network instead.
# """
# Base.delete!(tn::AbstractTensorNetwork, x) = (_ = pop!(tn, x); tn)

"""
    pop!(tn::AbstractTensorNetwork, tensor::Tensor)
    pop!(tn::AbstractTensorNetwork, i::Union{Symbol,AbstractVecOrTuple{Symbol}})

Remove and return the first tensor in `tn`` that satisfies _egality_ (i.e. `≡`or`===`) with `tensor`.

See also: [`push!`](@ref), [`delete!`](@ref).
"""
Base.pop!(tn::AbstractTensorNetwork, tensor::Tensor) = (delete!(tn, tensor); tensor)

"""
    pop!(tn::TensorNetwork, inds)

Remove and return the tensors that contain all the indices (i.e. `pop!` on each [`Tensor`](@ref) returned by `tensors(tn; contains=i)`).

See also: [`push!`](@ref), [`delete!`](@ref).
"""
function Base.pop!(tn::AbstractTensorNetwork, i::Union{Symbol,AbstractVecOrTuple{Symbol}})::Vector{Tensor}
    popping_tensors = tensors(tn; contains=i)
    foreach(Base.Fix1(delete!, tn), popping_tensors)
    return popping_tensors
end

"""
    replace!(tn::AbstractTensorNetwork, old => new...)
    replace(tn::AbstractTensorNetwork, old => new...)

Replace the element in `old` with the one in `new`. Depending on the types of `old` and `new`, the following behaviour is expected:

  - If `Symbol`s, it will correspond to a index renaming.
  - If `Tensor`s, first element that satisfies _egality_ (`≡` or `===`) will be replaced.
"""
Base.replace!(::AbstractTensorNetwork, ::Any...)

# rename index
function Base.replace!(tn::AbstractTensorNetwork, old_new::Pair{Symbol,Symbol})
    old, new = old_new
    old ∈ tn || throw(ArgumentError("index $old does not exist"))
    old == new && return tn
    new ∉ tn || throw(ArgumentError("index $new is already present"))
    # NOTE `copy` because collection underneath is mutated
    for old_tensor in copy(tensors(tn; contains=old))
        # NOTE do not `delete!` before `push!` as indices can be lost due to `tryprune!`
        new_tensor = replace(old_tensor, old_new)
        push_inner!(tn, new_tensor)
        delete_inner!(tn, old_tensor)
        handle(tn, ReplaceEffect(old_tensor => new_tensor))
    end
    tryprune!(tn, old)
    handle(tn, ReplaceEffect(old_new))
    return tn
end

# replace tensor
function Base.replace!(tn::AbstractTensorNetwork, old_new::Pair{<:Tensor,<:Tensor})
    old_tensor, new_tensor = old_new
    old_tensor === new_tensor && return tn

    issetequal(inds(new_tensor), inds(old_tensor)) || throw(ArgumentError("replacing tensor indices don't match"))

    push_inner!(tn, new_tensor)
    delete_inner!(tn, old_tensor)
    handle(tn, ReplaceEffect(old_new))

    return tn
end

# rename a collection of indices
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
        # TODO maybe do replacement manually and call `handle` once in the end?
        replace!(tn, tmp)

        # replace temporary names with new indices
        replace!(tn, [tmp[f] => t for (f, t) in zip(from, to)])
    end

    # return the final index mapping
    return tn
end

# replace tensor with a TensorNetwork
function Base.replace!(tn::AbstractTensorNetwork, old_new::Pair{<:Tensor,AbstractTensorNetwork})
    old, new = old_new
    issetequal(inds(new; set=:open), inds(old)) || throw(ArgumentError("indices don't match"))
    !isdisjoint(inds(new; set=:inner), inds(tn)) || throw(ArgumentError("overlapping inner indices"))

    # manually perform `append!(tn, new)` to avoid calling `handle` several times
    for tensor in tensors(new)
        push_inner!(tn, tensor)
    end
    delete_inner!(tn, old)
    handle(tn, ReplaceEffect(old_new))

    return tn
end

Base.replace!(tn::AbstractTensorNetwork) = tn
@inline Base.replace!(tn::T, old_new::P...) where {T<:AbstractTensorNetwork,P<:Pair} = replace!(tn, old_new)
@inline Base.replace!(tn::AbstractTensorNetwork, old_new::Dict) = replace!(tn, collect(old_new))

function Base.replace!(tn::AbstractTensorNetwork, old_new::Base.AbstractVecOrTuple{Pair})
    for pair in old_new
        replace!(tn, pair)
    end
    return tn
end

# TODO what if parallel neighbour indices?
"""
    contract!(tn::AbstractTensorNetwork, ind::Union{Symbol,AbstractVecOrTuple{Symbol}})

Contract in-place the index `ind` in the Tensor Network.
"""
function contract!(tn::AbstractTensorNetwork, inds)
    target_tensors = tensors(tn; intersects=inds)
    # TODO calling `contract` like this can give problems on large amount of tensors, because it doesn't call `einexpr`
    result_tensor = contract(target_tensors...; dims=inds)
    replace!(tn, target_tensors => result_tensor)
    return tn
end

# derived methods
Base.replace(tn::AbstractTensorNetwork, old_new::Pair...) = replace(tn, old_new)
Base.replace(tn::AbstractTensorNetwork, old_new) = replace!(copy(tn), old_new)

contract(tn::AbstractTensorNetwork, i; kwargs...) = contract!(copy(tn), i; kwargs...)

@inline Base.in(i::Symbol, tn::AbstractTensorNetwork) = hasind(tn, i)
@inline Base.in(tensor::Tensor, tn::AbstractTensorNetwork) = hastensor(tn, tensor)

Base.eltype(tn::AbstractTensorNetwork) = promote_type(eltype.(tensors(tn))...)

"""
    arrays(tn::AbstractTensorNetwork; kwargs...)

Return a list of the arrays of in the Tensor Network. It is equivalent to `parent.(tensors(tn; kwargs...))`.
"""
arrays(tn::AbstractTensorNetwork; kwargs...) = parent.(tensors(tn; kwargs...))

"""
    Base.collect(tn::AbstractTensorNetwork)

Return a list of the [`Tensor`](@ref)s in the Tensor Network. It is equivalent to `tensors(tn)`.
"""
Base.collect(tn::AbstractTensorNetwork) = tensors(tn)

# TODO should we deprecate this method?
function Base.getindex(tn::AbstractTensorNetwork, is::Symbol...; mul=1)
    first(Iterators.drop(tensors(tn; contains=is), mul - 1))
end

"""
    Base.similar(tn::AbstractTensorNetwork)

Return a copy of the `TensorNetwork` with all [`Tensor`](@ref)s replaced by their `similar` version.
"""
function Base.similar(tn::AbstractTensorNetwork)
    tn = copy(tn)
    for tensor in tensors(tn)
        replace!(tn, tensor => similar(tensor))
    end
    return tn
end

"""
    Base.zero(tn::AbstractTensorNetwork)

Return a copy of the `TensorNetwork` with all [`Tensor`](@ref)s replaced by their `zero` version.
"""
function Base.zero(tn::AbstractTensorNetwork)
    tn = copy(tn)
    for tensor in tensors(tn)
        replace!(tn, tensor => zero(tensor))
    end
    return tn
end

"""
    conj(tn::AbstractTensorNetwork)

Return a copy of the [`AbstractTensorNetwork`](@ref) with all tensors conjugated.

See also: [`conj!`](@ref).
"""
function Base.conj(tn::AbstractTensorNetwork)
    tn = copy(tn)
    replace!(tn, tensors(tn) .=> conj.(tensors(tn)))
    return tn
end

"""
    conj!(tn::AbstractTensorNetwork)

Conjugate all tensors in the [`AbstractTensorNetwork`](@ref) in-place.

See also: [`conj`](@ref).
"""
function Base.conj!(tn::AbstractTensorNetwork)
    foreach(conj!, tensors(tn))
    return tn
end

"""
    Graphs.neighbors(tn::AbstractTensorNetwork, tensor; open=true)

Return the neighboring [`Tensor`](@ref)s of `tensor` in the Tensor Network.
If `open=true`, the `tensor` itself is not included in the result.
"""
function Graphs.neighbors(tn::AbstractTensorNetwork, tensor::Tensor; open::Bool=true)
    @assert tensor ∈ tn "Tensor not found in TensorNetwork"
    neigh_tensors = mapreduce(∪, inds(tensor)) do index
        tensors(tn; intersects=index)
    end
    open && filter!(x -> x !== tensor, neigh_tensors)
    return neigh_tensors
end

"""
    Graphs.neighbors(tn::AbstractTensorNetwork, ind; open=true)

Return the neighboring indices of `ind` in the Tensor Network.
If `open=true`, the `ind` itself is not included in the result.
"""
function Graphs.neighbors(tn::AbstractTensorNetwork, i::Symbol; open::Bool=true)
    @assert i ∈ tn "Index $i not found in TensorNetwork"
    neigh_inds = mapreduce(inds, ∪, tensors(tn; intersects=i))
    open && filter!(x -> x !== i, neigh_inds)
    return neigh_inds
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

"""
    contract(tn::AbstractTensorNetwork; optimizer=Greedy(), path=einexpr(tn))

Contract a [`AbstractTensorNetwork`](@ref). If `path` is not specified, the contraction order will be computed by [`einexpr`](@ref).

See also: [`einexpr`](@ref), [`contract!`](@ref).
"""
contract(tn::AbstractTensorNetwork; kwargs...) = contract(sort_nt(values(kwargs)), tn)
contract(::@NamedTuple{}, tn::AbstractTensorNetwork) = contract((; path=einexpr(Greedy(), tn)), tn)

function contract(kwargs::NamedTuple{(:optimizer,)}, tn::AbstractTensorNetwork)
    path = einexpr(tn; optimizer=kwargs.optimizer)
    return contract(tn; path)
end

function contract(kwargs::NamedTuple{(:path,)}, tn::AbstractTensorNetwork)
    length(kwargs.path.args) == 0 && return tn[inds(kwargs.path)...]

    intermediates = map(subpath -> contract(tn; path=subpath), kwargs.path.args)
    return contract(intermediates...; dims=suminds(kwargs.path))
end

function Base.rand(::Type{T}, args...; kwargs...) where {T<:AbstractTensorNetwork}
    return rand(Random.default_rng(), T, args...; kwargs...)
end

function LinearAlgebra.svd!(tn::AbstractTensorNetwork; left_inds=Symbol[], right_inds=Symbol[], kwargs...)
    tensor = only(tensors(tn; contains=left_inds ∪ right_inds))
    U, s, Vt = svd(tensor; left_inds, right_inds, kwargs...)
    replace!(tn, tensor => [U, s, Vt])
    return tn
end

function LinearAlgebra.qr!(tn::AbstractTensorNetwork; left_inds=Symbol[], right_inds=Symbol[], kwargs...)
    tensor = only(tensors(tn; contains=left_inds ∪ right_inds))
    Q, R = qr(tensor; left_inds, right_inds, kwargs...)
    replace!(tn, tensor => [Q, R])
    return tn
end

function LinearAlgebra.lu!(tn::AbstractTensorNetwork; left_inds=Symbol[], right_inds=Symbol[], kwargs...)
    tensor = only(tensors(tn; contains=left_inds ∪ right_inds))
    L, U, P = lu(tensor; left_inds, right_inds, kwargs...)
    replace!(tn, tensor => [P, L, U])
    return tn
end

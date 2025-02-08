using Base: AbstractVecOrTuple
using Random
using EinExprs
using OMEinsum
using LinearAlgebra
using ScopedValues
using Serialization
using Graphs: Graphs

mutable struct CachedField{T}
    isvalid::Bool
    value::T
end

CachedField{T}() where {T} = CachedField{T}(false, T())

invalidate!(cf::CachedField) = cf.isvalid = false
function Base.get!(f, cf::CachedField)
    !cf.isvalid && (cf.value = f())
    cf.isvalid = true
    return cf.value
end

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
    TensorNetwork

Hypergraph of interconnected tensors, representing a multilinear equation aka Tensor Network.
Vertices represent tensors and edges, tensor indices.
"""
struct TensorNetwork <: AbstractTensorNetwork
    indexmap::Dict{Symbol,Vector{Tensor}}
    tensormap::IdDict{Tensor,Vector{Symbol}}

    sorted_tensors::CachedField{Vector{Tensor}}
    unsafe::Ref{Union{Nothing,UnsafeScope}}

    # TODO: Find a way to remove the `unsafe` keyword argument from the constructor
    function TensorNetwork(tensors; unsafe::Union{Nothing,UnsafeScope}=nothing)
        tensormap = IdDict{Tensor,Vector{Symbol}}(tensor => collect(inds(tensor)) for tensor in tensors)

        indexmap = reduce(tensors; init=Dict{Symbol,Vector{Tensor}}()) do dict, tensor
            for index in inds(tensor)
                # TODO use lambda? `Tensor[]` might be reused
                push!(get!(dict, index, Tensor[]), tensor)
            end
            dict
        end

        # Check index size consistency if not inside an `UnsafeScope`
        if isnothing(unsafe)
            for ind in keys(indexmap)
                dims = map(tensor -> size(tensor, ind), indexmap[ind])
                length(unique(dims)) == 1 ||
                    throw(DimensionMismatch("Index $(ind) has inconsistent dimension: $(dims)"))
            end
        end

        return new(indexmap, tensormap, CachedField{Vector{Tensor}}(), Ref{Union{Nothing,UnsafeScope}}(unsafe))
    end
end

TensorNetwork() = TensorNetwork(Tensor[])
TensorNetwork(tn::TensorNetwork) = tn

get_unsafe_scope(tn::AbstractTensorNetwork) = TensorNetwork(tn).unsafe[]
function set_unsafe_scope!(tn::AbstractTensorNetwork, uc::Union{Nothing,UnsafeScope})
    TensorNetwork(tn).unsafe[] = uc
    return tn
end

"""
    copy(tn::TensorNetwork)

Return a shallow copy of a [`TensorNetwork`](@ref); i.e. changes to the copied `TensorNetwork` won't affect the original one, but changes to the tensors will.
"""
function Base.copy(tn::TensorNetwork)
    new_tn = TensorNetwork(tensors(tn); unsafe=get_unsafe_scope(tn))

    if !isnothing(get_unsafe_scope(tn))
        push!(get_unsafe_scope(tn).refs, WeakRef(new_tn)) # Register the new copy to the proper UnsafeScope
    end

    return new_tn
end

Base.similar(tn::TensorNetwork) = TensorNetwork(similar.(tensors(tn)))
Base.zero(tn::TensorNetwork) = TensorNetwork(zero.(tensors(tn)))

Base.summary(io::IO, tn::AbstractTensorNetwork) = print(io, "$(ntensors(tn))-tensors TensorNetwork")
function Base.show(io::IO, tn::T) where {T<:AbstractTensorNetwork}
    return print(io, "$T (#tensors=$(ntensors(tn)), #inds=$(ninds(tn)))")
end

Base.:(==)(a::AbstractTensorNetwork, b::AbstractTensorNetwork) = all(splat(==), zip(tensors(a), tensors(b)))
function Base.isapprox(a::AbstractTensorNetwork, b::AbstractTensorNetwork; kwargs...)
    return all(((x, y),) -> isapprox(x, y; kwargs...), zip(tensors(a), tensors(b)))
end

Base.eltype(tn::AbstractTensorNetwork) = promote_type(eltype.(tensors(tn))...)

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

"""
    inds(tn::AbstractTensorNetwork, set = :all)

Return the names of the indices in the [`AbstractTensorNetwork`](@ref).

# Keyword Arguments

  - `set`

      + `:all` (default) All indices.
      + `:open` Indices only mentioned in one tensor.
      + `:inner` Indices mentioned at least twice.
      + `:hyper` Indices mentioned at least in three tensors.
      + `:parallelto` Indices parallel to `i` in the graph (`i` included).
"""
function inds end

inds(tn::AbstractTensorNetwork; kwargs...) = inds(sort_nt(values(kwargs)), tn)
inds(::@NamedTuple{}, tn::AbstractTensorNetwork) = inds((; set=:all), tn)

function inds(kwargs::NamedTuple{(:set,)}, tn::AbstractTensorNetwork)
    tn = TensorNetwork(tn)
    if kwargs.set === :all
        collect(keys(tn.indexmap))
    elseif kwargs.set === :open
        map(first, Iterators.filter(((_, v),) -> length(v) == 1, tn.indexmap))
    elseif kwargs.set === :inner
        map(first, Iterators.filter(((_, v),) -> length(v) >= 2, tn.indexmap))
    elseif kwargs.set === :hyper
        map(first, Iterators.filter(((_, v),) -> length(v) >= 3, tn.indexmap))
    else
        throw(ArgumentError("""
          Unknown query: set=$(kwargs.set)
          Possible options are:
            - :all (default)
            - :open
            - :inner
            - :hyper
          For `AbstractQuantum`, the following are also available:
            - :physical
            - :virtual
          """))
    end
end

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
ninds(tn::AbstractTensorNetwork; kwargs...) = ninds(values(kwargs), tn)
ninds(::@NamedTuple{}, tn::AbstractTensorNetwork) = length(TensorNetwork(tn).indexmap)
ninds(kwargs::NamedTuple, tn::AbstractTensorNetwork) = length(inds(kwargs, tn))

"""
    tensors(tn::AbstractTensorNetwork)

Return a list of the `Tensor`s in the [`AbstractTensorNetwork`](@ref).

# Implementation details

  - As the tensors of a [`AbstractTensorNetwork`](@ref) are stored as keys of the `.tensormap` dictionary and it uses `objectid` as hash, order is not stable so it sorts for repeated evaluations.
"""
function tensors end

tensors(tn::AbstractTensorNetwork; kwargs...) = tensors(sort_nt(values(kwargs)), tn)

function tensors(::@NamedTuple{}, tn::AbstractTensorNetwork)
    tn = TensorNetwork(tn)
    get!(tn.sorted_tensors) do
        sort!(collect(keys(tn.tensormap)); by=sort ∘ collect ∘ inds)
    end
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
ntensors(::@NamedTuple{}, tn::AbstractTensorNetwork) = length(TensorNetwork(tn).tensormap)
ntensors(kwargs::NamedTuple, tn::AbstractTensorNetwork) = length(tensors(kwargs, tn))

# TODO move to `tensors` method
function Base.getindex(tn::AbstractTensorNetwork, is::Symbol...; mul::Int=1)
    tn = TensorNetwork(tn)
    return first(Iterators.drop(Iterators.filter(Base.Fix1(issetequal, is) ∘ inds, tn.indexmap[first(is)]), mul - 1))
end

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
    in(tensor::Tensor, tn::TensorNetwork)
    in(index::Symbol, tn::TensorNetwork)

Return `true` if there is a `Tensor` in `tn` for which `==` evaluates to `true`.
This method is equivalent to `tensor ∈ tensors(tn)` code, but it's faster on large amount of tensors.
"""
Base.in(tensor::Tensor, tn::AbstractTensorNetwork) = tensor ∈ keys(TensorNetwork(tn).tensormap)
Base.in(index::Symbol, tn::AbstractTensorNetwork) = index ∈ keys(TensorNetwork(tn).indexmap)

Base.size(tn::AbstractTensorNetwork, args...) = size(TensorNetwork(tn), args...)
"""
    size(tn::AbstractTensorNetwork)
    size(tn::AbstractTensorNetwork, index)

Return a mapping from indices to their dimensionalities.

If `index` is set, return the dimensionality of `index`. This is equivalent to `size(tn)[index]`.
"""
function Base.size(tn::AbstractTensorNetwork)
    return Dict{Symbol,Int}(index => size(tn, index) for index in keys(TensorNetwork(tn).indexmap))
end
Base.size(tn::AbstractTensorNetwork, index::Symbol) = size(first(TensorNetwork(tn).indexmap[index]), index)

function __check_index_sizes(tn)
    # Iterate through each index in the indexmap
    for (index, tensors) in tn.indexmap
        # Get the size of the first tensor for this index
        reference_size = size(tensors[1], index)

        # Compare the size of each subsequent tensor for this index
        for tensor in tensors
            if size(tensor, index) != reference_size
                return false
            end
        end
    end

    return true
end

Base.in(tn::TensorNetwork, uc::UnsafeScope) = tn ∈ values(uc)

macro unsafe_region(tn_sym, block)
    return esc(
        quote
            local old = copy($tn_sym)

            # Create a new UnsafeScope and set it to the current tn
            local _uc = Tenet.UnsafeScope()
            Tenet.set_unsafe_scope!($tn_sym, _uc)

            # Register the tensor network in the UnsafeScope
            push!(Tenet.get_unsafe_scope($tn_sym).refs, WeakRef($tn_sym))

            e = nothing
            try
                $(block) # Execute the user-provided block
            catch e
                $(tn_sym) = old # Restore the original tensor network in case of an exception
                rethrow(e)
            finally
                if isnothing(e)
                    # Perform checks of registered tensor networks
                    for ref in Tenet.get_unsafe_scope($tn_sym).refs
                        tn = ref.value
                        if !isnothing(tn) && tn ∈ values(Tenet.get_unsafe_scope($tn_sym))
                            if !Tenet.__check_index_sizes(tn)
                                $(tn_sym) = old

                                # Set `unsafe` field to `nothing`
                                Tenet.set_unsafe_scope!($tn_sym, nothing)

                                throw(DimensionMismatch("Inconsistent size of indices"))
                            end
                        end
                    end
                end
            end
        end,
    )
end

"""
    push!(tn::AbstractTensorNetwork, tensor::Tensor)

Add a new `tensor` to the Tensor Network.

See also: [`append!`](@ref), [`pop!`](@ref).
"""
function Base.push!(tn::AbstractTensorNetwork, tensor::Tensor)
    tn = TensorNetwork(tn)
    tensor ∈ keys(tn.tensormap) && return tn

    # Check index sizes if there isn't an active `UnsafeScope` in the Tensor Network
    if isnothing(get_unsafe_scope(tn))
        for i in Iterators.filter(i -> size(tn, i) != size(tensor, i), inds(tensor) ∩ inds(tn))
            throw(
                DimensionMismatch("size(tensor,$i)=$(size(tensor,i)) but should be equal to size(tn,$i)=$(size(tn,i))")
            )
        end
    end

    tn.tensormap[tensor] = collect(inds(tensor))
    for index in unique(inds(tensor))
        push!(get!(tn.indexmap, index, Tensor[]), tensor)
    end

    invalidate!(tn.sorted_tensors)

    return tn
end

"""
    append!(tn::TensorNetwork, tensors::AbstractVecOrTuple{<:Tensor})

Add a list of tensors to a `TensorNetwork`.

See also: [`push!`](@ref), [`merge!`](@ref).
"""
Base.append!(tn::AbstractTensorNetwork, tensors) = (foreach(Base.Fix1(push!, tn), tensors); tn)

"""
    pop!(tn::TensorNetwork, tensor::Tensor)
    pop!(tn::TensorNetwork, i::Union{Symbol,AbstractVecOrTuple{Symbol}})

Remove a tensor from the Tensor Network and returns it. If a `Tensor` is passed, then the first tensor satisfies _egality_ (i.e. `≡` or `===`) will be removed.
If a `Symbol` or a list of `Symbol`s is passed, then remove and return the tensors that contain all the indices.

See also: [`push!`](@ref), [`delete!`](@ref).
"""
Base.pop!(tn::AbstractTensorNetwork, tensor::Tensor) = (delete!(TensorNetwork(tn), tensor); tensor)
Base.pop!(tn::AbstractTensorNetwork, i::Symbol) = pop!(TensorNetwork(tn), (i,))

function Base.pop!(tn::AbstractTensorNetwork, i::AbstractVecOrTuple{Symbol})::Vector{Tensor}
    tensorlist = tensors(tn; contains=i)
    for tensor in tensorlist
        _ = pop!(tn, tensor)
    end

    return tensorlist
end

"""
    delete!(tn::TensorNetwork, x)

Like [`pop!`](@ref) but return the [`TensorNetwork`](@ref) instead.
"""
Base.delete!(tn::AbstractTensorNetwork, x) = (_ = pop!(tn, x); tn)

function tryprune!(tn::AbstractTensorNetwork, i::Symbol)
    if i ∈ tn
        tn = TensorNetwork(tn)
        isempty(tn.indexmap[i]) && delete!(tn.indexmap, i)
    end
    return nothing
end

function Base.delete!(tn::TensorNetwork, tensor::Tensor)
    for index in unique(inds(tensor))
        filter!(Base.Fix1(!==, tensor), tn.indexmap[index])
        tryprune!(tn, index)
    end
    delete!(tn.tensormap, tensor)

    invalidate!(tn.sorted_tensors)

    return tn
end

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

function Base.replace!(tn::AbstractTensorNetwork, old_new::Pair{Symbol,Symbol})
    orig_tn = tn
    tn = TensorNetwork(tn)
    old, new = old_new
    old ∈ tn || throw(ArgumentError("index $old does not exist"))
    old == new && return tn
    new ∉ tn || throw(ArgumentError("index $new is already present"))
    # NOTE `copy` because collection underneath is mutated
    for tensor in copy(tensors(tn; contains=old))
        # NOTE do not `delete!` before `push!` as indices can be lost due to `tryprune!`
        push!(tn, replace(tensor, old_new))
        delete!(tn, tensor)
    end
    tryprune!(tn, old)
    return orig_tn
end

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

function Base.replace!(tn::AbstractTensorNetwork, pair::Pair{<:Tensor,<:Tensor})
    tn = TensorNetwork(tn)
    old_tensor, new_tensor = pair

    old_tensor === new_tensor && return tn

    issetequal(inds(new_tensor), inds(old_tensor)) || throw(ArgumentError("replacing tensor indices don't match"))

    push!(tn, new_tensor)
    delete!(tn, old_tensor)

    return tn
end

function Base.replace!(tn::AbstractTensorNetwork, old_new::Pair{<:Tensor,<:TensorNetwork})
    tn = TensorNetwork(tn)
    old, new = old_new
    issetequal(inds(new; set=:open), inds(old)) || throw(ArgumentError("indices don't match"))

    # rename internal indices so there is no accidental hyperedge
    replace!(new, [index => Symbol(uuid4()) for index in filter(∈(inds(tn)), inds(new; set=:inner))])

    merge!(tn, new)
    delete!(tn, old)

    return tn
end

"""
    resetinds!(tn::AbstractTensorNetwork; init::Int=1)

Rename all indices in the `TensorNetwork` to a new set of indices starting from `init`th Unicode character.
"""
function resetinds!(tn::AbstractTensorNetwork; init::Int=1)
    mapping = resetinds!(Val(:return_mapping), tn; init=init)
    return replace!(tn, mapping)
end

function resetinds!(::Val{:return_mapping}, tn::AbstractTensorNetwork; init::Int=1)
    gen = IndexCounter(init)
    return Dict{Symbol,Symbol}([i => nextindex!(gen) for i in inds(tn)])
end

"""
    merge!(self::TensorNetwork, others::TensorNetwork...)
    merge(self::TensorNetwork, others::TensorNetwork...)

Fuse various [`TensorNetwork`](@ref)s into one.

See also: [`append!`](@ref).
"""
Base.merge!(self::TensorNetwork, other::TensorNetwork) = append!(self, tensors(other))
Base.merge!(self::TensorNetwork, others::TensorNetwork...) = foldl(merge!, others; init=self)
Base.merge(self::AbstractTensorNetwork, others::AbstractTensorNetwork...) = merge!(copy(self), others...)

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
    slice!(tn::AbstractTensorNetwork, index::Symbol, i)

In-place projection of `index` on dimension `i`.

See also: [`selectdim`](@ref), [`view`](@ref).
"""
function slice!(tn::AbstractTensorNetwork, label::Symbol, i)
    for tensor in pop!(TensorNetwork(tn), label)
        push!(TensorNetwork(tn), selectdim(tensor, label, i))
    end

    return tn
end

"""
    selectdim(tn::AbstractTensorNetwork, index::Symbol, i)

Return a copy of the [`AbstractTensorNetwork`](@ref) where `index` has been projected to dimension `i`.

See also: [`view`](@ref), [`slice!`](@ref).
"""
Base.selectdim(tn::AbstractTensorNetwork, index::Symbol, i) = @view tn[index => i]

"""
    view(tn::AbstractTensorNetwork, index => i...)

Return a copy of the [`AbstractTensorNetwork`](@ref) where each `index` has been projected to dimension `i`.
It is equivalent to a recursive call of [`selectdim`](@ref).

See also: [`selectdim`](@ref), [`slice!`](@ref).
"""
function Base.view(tn::AbstractTensorNetwork, slices::Pair{Symbol}...)
    tn = copy(tn)

    for (label, i) in slices
        slice!(tn, label, i)
    end

    return tn
end

"""
    fuse!(tn::AbstractTensorNetwork, i::Symbol)

Group indices parallel to `i` and reshape the tensors accordingly.
"""
function fuse!(tn::AbstractTensorNetwork, i)
    parinds = filter!(!=(i), inds(tn; parallelto=i))
    length(parinds) == 0 && return tn

    parinds = (i,) ∪ parinds
    newtensors = map(Base.Fix2(fuse, parinds), pop!(tn, parinds))

    append!(tn, newtensors)

    return tn
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
    contract(tn::AbstractTensorNetwork; path=einexpr(tn))

Contract a [`AbstractTensorNetwork`](@ref). If `path` is not specified, the contraction order will be computed by [`einexpr`](@ref).

See also: [`einexpr`](@ref), [`contract!`](@ref).
"""
contract(tn::AbstractTensorNetwork; kwargs...) = contract(sort_nt(values(kwargs)), tn)
contract(::@NamedTuple{}, tn::AbstractTensorNetwork) = contract((; path=einexpr(tn)), tn)
function contract(kwargs::NamedTuple{(:path,)}, tn::AbstractTensorNetwork)
    length(kwargs.path.args) == 0 && return tn[inds(kwargs.path)...]

    intermediates = map(subpath -> contract(tn; path=subpath), kwargs.path.args)
    return contract(intermediates...; dims=suminds(kwargs.path))
end

"""
    contract!(tn::AbstractTensorNetwork; path=einexpr(tn))

Same as [`contract`](@ref) but in-place.

See also: [`einexpr`](@ref).
"""
contract!(tn::AbstractTensorNetwork; kwargs...) = contract!(sort_nt(values(kwargs)), tn)

# TODO sequence of indices?
# TODO what if parallel neighbour indices?
"""
    contract!(tn::AbstractTensorNetwork, index)

In-place contraction of tensors connected to `index`.

See also: [`contract`](@ref).
"""
function contract!(tn::AbstractTensorNetwork, i)
    _tensors = sort!(tensors(tn; intersects=i); by=length)
    tensor = contract(TensorNetwork(_tensors))
    _tn = TensorNetwork(tn)
    delete!(_tn, i)
    push!(_tn, tensor)
    return tn
end
contract!(tn::AbstractTensorNetwork, i::Symbol) = contract!(tn, [i])
contract(tn::AbstractTensorNetwork, i; kwargs...) = contract!(copy(tn), i; kwargs...)

function contract!(tn::AbstractTensorNetwork, t::Tensor; kwargs...)
    tn = TensorNetwork(tn)
    push!(tn, t)
    return contract(tn; kwargs...)
end
contract!(t::Tensor, tn::AbstractTensorNetwork; kwargs...) = contract!(tn, t; kwargs...)
contract(t::Tensor, tn::AbstractTensorNetwork; kwargs...) = contract(tn, t; kwargs...)

function LinearAlgebra.svd!(tn::AbstractTensorNetwork; left_inds=Symbol[], right_inds=Symbol[], kwargs...)
    tensor = tn[left_inds ∪ right_inds...]
    U, s, Vt = svd(tensor; left_inds, right_inds, kwargs...)
    replace!(tn, tensor => TensorNetwork([U, s, Vt]))
    return tn
end

function LinearAlgebra.qr!(tn::AbstractTensorNetwork; left_inds=Symbol[], right_inds=Symbol[], kwargs...)
    tensor = only(tensors(tn; contains=left_inds ∪ right_inds))
    Q, R = qr(tensor; left_inds, right_inds, kwargs...)
    replace!(tn, tensor => TensorNetwork([Q, R]))
    return tn
end

function LinearAlgebra.lu!(tn::AbstractTensorNetwork; left_inds=Symbol[], right_inds=Symbol[], kwargs...)
    tensor = tn[left_inds ∪ right_inds...]
    L, U, P = lu(tensor; left_inds, right_inds, kwargs...)
    replace!(tn, tensor => TensorNetwork([P, L, U]))
    return tn
end

"""
    rand(TensorNetwork, n::Integer, regularity::Integer; out = 0, dim = 2:9, seed = nothing, globalind = false)

Generate a random tensor network.

# Arguments

  - `n` Number of tensors.
  - `regularity` Average number of indices per tensor.
  - `out` Number of open indices.
  - `dim` Range of dimension sizes.
  - `seed` If not `nothing`, seed random generator with this value.
  - `globalind` Add a global 'broadcast' dimension to every tensor.
"""
function Base.rand(
    rng::Random.AbstractRNG,
    ::Type{TensorNetwork},
    n::Integer,
    regularity::Integer;
    out=0,
    dim=2:9,
    seed=nothing,
    globalind=false,
    eltype=Float64,
)
    !isnothing(seed) && Random.seed!(rng, seed)

    inds = letter.(randperm(n * regularity ÷ 2 + out))
    size_dict = Dict(ind => rand(dim) for ind in inds)

    outer_inds = collect(Iterators.take(inds, out))
    inner_inds = collect(Iterators.drop(inds, out))

    candidate_inds = shuffle(
        collect(Iterators.flatten([outer_inds, Iterators.flatten(Iterators.repeated(inner_inds, 2))]))
    )

    inputs = map(x -> [x], Iterators.take(candidate_inds, n))

    for ind in Iterators.drop(candidate_inds, n)
        i = rand(1:n)
        while ind in inputs[i]
            i = rand(1:n)
        end

        push!(inputs[i], ind)
    end

    if globalind
        ninds = length(size_dict)
        ind = letter(ninds + 1)
        size_dict[ind] = rand(dim)
        push!(outer_inds, ind)
        push!.(inputs, (ind,))
    end

    tensors = Tensor[Tensor(rand(eltype, [size_dict[ind] for ind in input]...), tuple(input...)) for input in inputs]
    return TensorNetwork(tensors)
end

function Base.rand(::Type{TensorNetwork}, n::Integer, regularity::Integer; kwargs...)
    return rand(Random.default_rng(), TensorNetwork, n, regularity; kwargs...)
end

function Base.rand(::Type{T}, args...; kwargs...) where {T<:AbstractTensorNetwork}
    return rand(Random.default_rng(), T, args...; kwargs...)
end

function Serialization.serialize(s::AbstractSerializer, obj::TensorNetwork)
    Serialization.writetag(s.io, Serialization.OBJECT_TAG)
    serialize(s, TensorNetwork)
    return serialize(s, tensors(obj))
end

function Serialization.deserialize(s::AbstractSerializer, ::Type{TensorNetwork})
    ts = deserialize(s)
    return TensorNetwork(ts)
end

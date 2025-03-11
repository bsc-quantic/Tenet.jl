using Random
using EinExprs
using LinearAlgebra
using ScopedValues
using Serialization
using Graphs: Graphs
using ValSplit

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
                # avoid multiple references to the same tensor
                if isnothing(findfirst(x -> x === tensor, get!(dict, index, Tensor[])))
                    push!(dict[index], tensor)
                end
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

# UnsafeScope
get_unsafe_scope(tn::TensorNetwork) = TensorNetwork(tn).unsafe[]
set_unsafe_scope!(tn::TensorNetwork, uc::Union{Nothing,UnsafeScope}) = TensorNetwork(tn).unsafe[] = uc

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
        sort!(collect(keys(tn.tensormap)); by=sort ∘ collect ∘ vinds)
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
                            if !Tenet.checksizes(tn)
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

# "Tensor Network" interface
trait(::TensorNetworkInterface, ::TensorNetwork) = IsTensorNetwork()

## required methods
function tensors(::@NamedTuple{}, tn::TensorNetwork)
    get!(tn.sorted_tensors) do
        sort!(collect(keys(tn.tensormap)); by=sort ∘ collect ∘ vinds)
    end
end

function Base.copy(tn::TensorNetwork)
    new_tn = TensorNetwork(tensors(tn); unsafe=get_unsafe_scope(tn))

    if !isnothing(get_unsafe_scope(tn))
        push!(get_unsafe_scope(tn).refs, WeakRef(new_tn)) # Register the new copy to the proper UnsafeScope
    end

    return new_tn
end

## optional methods
hastensor(tn::TensorNetwork, tensor::Tensor) = tensor ∈ keys(tn.tensormap)
hasind(tn::TensorNetwork, index::Symbol) = index ∈ keys(tn.indexmap)

# here for performance reasons
# on my macbook M1 pro with a TensorNetwork of 1000 tensors, 2000 indices
# - `ninds` takes 50 ns
# - `length(inds(tn))` takes 90 ns
ntensors(::@NamedTuple{}, tn::TensorNetwork) = length(TensorNetwork(tn).tensormap)

# here for performance reasons
# on my macbook M1 pro with a TensorNetwork of 100 tensors, 250 indices
# - `ninds` takes 50 ns
# - `length(inds(tn))` takes 1 μs
ninds(::@NamedTuple{}, tn::TensorNetwork) = length(TensorNetwork(tn).indexmap)

# here for performance reasons
# NOTE if needed we can use `Base.ImmutableDict`
function Base.size(tn::TensorNetwork)
    return Dict{Symbol,Int}(index => size(tn, index) for index in keys(tn.indexmap))
end
Base.size(tn::TensorNetwork, index::Symbol) = size(first(tn.indexmap[index]), index)

## keyword methods
@valsplit function inds(Val(kwargs::@NamedTuple{set::Symbol}), tn::TensorNetwork)
    throw(ArgumentError("""
        Unknown query: set=$(kwargs.set)
        Possible options are:
          - :all (default)
          - :open
          - :inner
          - :hyper
        """))
end

inds(::Val{(; set = :all)}, tn::TensorNetwork) = collect(keys(tn.indexmap))

function inds(::Val{(; set = :open)}, tn::TensorNetwork)
    # optimized by just considering the inds whose `tn.indexmap[ind]` has length == 1
    filter(tn.indexmap) do (ind, vs)
        length(vs) > 1 && return false
        count(==(ind), Iterators.flatmap(Tenet.vinds, vs)) == 1
    end |> keys |> collect
end

function inds(::Val{(; set = :inner)}, tn::TensorNetwork)
    # optimized by preadding the inds whose `tn.indexmap[ind]` has length >= 2
    filter(tn.indexmap) do (ind, vs)
        length(vs) >= 2 && return true
        count(==(ind), Iterators.flatmap(Tenet.vinds, vs)) >= 2
    end |> keys |> collect
end

function inds(::Val{(; set = :hyper)}, tn::TensorNetwork)
    # optimized by preadding the inds whose `tn.indexmap[ind]` has length >= 3
    filter(tn.indexmap) do (ind, vs)
        length(vs) >= 3 && return true
        count(==(ind), Iterators.flatmap(Tenet.vinds, vs)) >= 3
    end |> keys |> collect
end

tensors(kwargs::@NamedTuple{contains::Symbol}, tn::TensorNetwork) = copy(TensorNetwork(tn).indexmap[kwargs.contains])
function tensors(kwargs::NamedTuple{(:contains,)}, tn::TensorNetwork)
    target_tensors = tensors(tn; contains=first(kwargs.contains))
    filter!(target_tensors) do tensor
        kwargs.contains ⊆ inds(tensor)
    end
    return target_tensors
end

## mutating methods
function push_inner!(tn::TensorNetwork, tensor::Tensor)
    hastensor(tn, tensor) && return tn

    # check index sizes if there isn't an active `UnsafeScope` in the Tensor Network
    if isnothing(get_unsafe_scope(tn))
        for i in Iterators.filter(i -> size(tn, i) != size(tensor, i), inds(tensor) ∩ inds(tn))
            throw(
                DimensionMismatch("size(tensor,$i)=$(size(tensor,i)) but should be equal to size(tn,$i)=$(size(tn,i))")
            )
        end
    end

    # do the actual push
    tn.tensormap[tensor] = collect(inds(tensor))
    for index in unique(inds(tensor))
        push!(get!(tn.indexmap, index, Tensor[]), tensor)
    end

    # tensors have changed, invalidate cache and reconstruct on next `tensors` call
    invalidate!(tn.sorted_tensors)

    return tn
end

function delete_inner!(tn::TensorNetwork, tensor::Tensor)
    # do the actual delete
    for index in unique(inds(tensor))
        filter!(Base.Fix1(!==, tensor), tn.indexmap[index])
        tryprune!(tn, index)
    end
    delete!(tn.tensormap, tensor)

    # tensors have changed, invalidate cache and reconstruct on next `tensors` call
    invalidate!(tn.sorted_tensors)

    return tn
end

function tryprune!(tn::TensorNetwork, i::Symbol)
    if i ∈ tn
        tn = TensorNetwork(tn)
        isempty(tn.indexmap[i]) && delete!(tn.indexmap, i)
    end
    return nothing
end

# required for `canhandle` and thus, `push!`, `delete!` and `replace!`  to work
# TODO might be a good idea to move `push_inner!` and `delete_inner!` to this
handle!(::TensorNetwork, @nospecialize(e::PushEffect{T})) where {T<:Tensor} = nothing
handle!(::TensorNetwork, @nospecialize(e::DeleteEffect{T})) where {T<:Tensor} = nothing
handle!(::TensorNetwork, @nospecialize(e::ReplaceEffect{Pair{Symbol,Symbol}})) = nothing
handle!(::TensorNetwork, @nospecialize(e::ReplaceEffect{Pair{A,B}})) where {A<:Tensor,B<:Tensor} = nothing
handle!(::TensorNetwork, @nospecialize(e::ReplaceEffect{Pair{T,TensorNetwork}})) where {T} = nothing
handle!(::TensorNetwork, @nospecialize(e::ReplaceEffect{Pair{Vector{A},B}})) where {A<:Tensor,B<:Tensor} = nothing

## derived methods
Base.summary(io::IO, tn::TensorNetwork) = print(io, "$(ntensors(tn))-tensors TensorNetwork")
function Base.show(io::IO, tn::T) where {T<:TensorNetwork}
    return print(io, "$T (#tensors=$(ntensors(tn)), #inds=$(ninds(tn)))")
end

Base.:(==)(a::TensorNetwork, b::TensorNetwork) = all(splat(==), zip(tensors(a), tensors(b)))
function Base.isapprox(a::TensorNetwork, b::TensorNetwork; kwargs...)
    return all(((x, y),) -> isapprox(x, y; kwargs...), zip(tensors(a), tensors(b)))
end

# TODO move it to interface?
"""
    slice!(tn::AbstractTensorNetwork, index::Symbol, i)

In-place projection of `index` on dimension `i`.

See also: [`selectdim`](@ref), [`view`](@ref).
"""
function slice!(tn::TensorNetwork, label::Symbol, i)
    for tensor in pop!(TensorNetwork(tn), label)
        push!(TensorNetwork(tn), selectdim(tensor, label, i))
    end

    return tn
end

# TODO move it to interface?
"""
    selectdim(tn::AbstractTensorNetwork, index::Symbol, i)

Return a copy of the [`AbstractTensorNetwork`](@ref) where `index` has been projected to dimension `i`.

See also: [`view`](@ref), [`slice!`](@ref).
"""
Base.selectdim(tn::TensorNetwork, index::Symbol, i) = @view tn[index => i]

# TODO move it to interface?
"""
    view(tn::AbstractTensorNetwork, index => i...)

Return a copy of the [`AbstractTensorNetwork`](@ref) where each `index` has been projected to dimension `i`.
It is equivalent to a recursive call of [`selectdim`](@ref).

See also: [`selectdim`](@ref), [`slice!`](@ref).
"""
function Base.view(tn::TensorNetwork, slices::Pair{Symbol}...)
    tn = copy(tn)

    for (label, i) in slices
        slice!(tn, label, i)
    end

    return tn
end

# TODO move it to interface?
"""
    fuse!(tn::TensorNetwork, i::Symbol)

Group indices parallel to `i` and reshape the tensors accordingly.
"""
function fuse!(tn::TensorNetwork, i)
    parinds = filter!(!=(i), inds(tn; parallelto=i))
    length(parinds) == 0 && return tn

    parinds = (i,) ∪ parinds
    newtensors = map(Base.Fix2(fuse, parinds), pop!(tn, parinds))

    append!(tn, newtensors)

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

function Serialization.serialize(s::AbstractSerializer, obj::TensorNetwork)
    Serialization.writetag(s.io, Serialization.OBJECT_TAG)
    serialize(s, TensorNetwork)
    return serialize(s, tensors(obj))
end

function Serialization.deserialize(s::AbstractSerializer, ::Type{TensorNetwork})
    ts = deserialize(s)
    return TensorNetwork(ts)
end

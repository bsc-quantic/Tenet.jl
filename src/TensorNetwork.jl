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

get_unsafe_scope(tn::TensorNetwork) = tn.unsafe[]
set_unsafe_scope!(tn::TensorNetwork, uc::Union{Nothing,UnsafeScope}) = tn.unsafe[] = uc

function checksizes(tn)
    # TODO better interface this: dispatch on Interface?
    tn = unwrap(TensorNetworkInterface(), tn)

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

Base.similar(tn::TensorNetwork) = TensorNetwork(similar.(tensors(tn)))
Base.zero(tn::TensorNetwork) = TensorNetwork(zero.(tensors(tn)))

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

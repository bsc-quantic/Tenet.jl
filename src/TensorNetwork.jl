using Base: AbstractVecOrTuple
using Random
using EinExprs
using OMEinsum
using LinearAlgebra
using ScopedValues
using Serialization
using Graphs: Graphs

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

# `WrapsTensorNetwork`-trait
# returns `No` because we `TensorNetwork` methods will already correctly dispatch correctly
Wraps(::Type{TensorNetwork}, _) = No()

function get_unsafe_scope(tn::TensorNetwork)
    if Wraps(TensorNetwork, tn) isa Yes
        return TensorNetwork(tn).unsafe[]
    else
        error("UnsafeScope is only available for TensorNetwork or wrappers")
    end
end

function set_unsafe_scope!(tn::TensorNetwork, uc::Union{Nothing,UnsafeScope})
    if Wraps(TensorNetwork, tn) isa Yes
        TensorNetwork(tn).unsafe[] = uc
    else
        error("UnsafeScope is only available for TensorNetwork or wrappers")
    end
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

Base.summary(io::IO, tn::TensorNetwork) = print(io, "$(ntensors(tn))-tensors TensorNetwork")
function Base.show(io::IO, tn::T) where {T<:TensorNetwork}
    return print(io, "$T (#tensors=$(ntensors(tn)), #inds=$(ninds(tn)))")
end

Base.:(==)(a::TensorNetwork, b::TensorNetwork) = all(splat(==), zip(tensors(a), tensors(b)))
function Base.isapprox(a::TensorNetwork, b::TensorNetwork; kwargs...)
    return all(((x, y),) -> isapprox(x, y; kwargs...), zip(tensors(a), tensors(b)))
end

inds_set(tn, ::Val{:all}) = collect(keys(tn.indexmap))
inds_set(tn, ::Val{:open}) = map(first, Iterators.filter((_, v) -> length(v) == 1, tn.indexmap))
inds_set(tn, ::Val{:inner}) = map(first, Iterators.filter((_, v) -> length(v) >= 2, tn.indexmap))
inds_set(tn, ::Val{:hyper}) = map(first, Iterators.filter((_, v) -> length(v) >= 3, tn.indexmap))

# here for performance reasons
# on my macbook M1 pro with a TensorNetwork of 100 tensors, 250 indices
# - `ninds` takes 50 ns
# - `length(inds(tn))` takes 1 μs
ninds(::@NamedTuple{}, tn::TensorNetwork) = length(TensorNetwork(tn).indexmap)

function tensors(::@NamedTuple{}, tn::TensorNetwork)
    get!(tn.sorted_tensors) do
        sort!(collect(keys(tn.tensormap)); by=sort ∘ collect ∘ inds)
    end
end

# here for performance reasons
# on my macbook M1 pro with a TensorNetwork of 1000 tensors, 2000 indices
# - `ninds` takes 50 ns
# - `length(inds(tn))` takes 90 ns
ntensors(::@NamedTuple{}, tn::TensorNetwork) = length(TensorNetwork(tn).tensormap)

# TODO move to `tensors` method
function Base.getindex(tn::TensorNetwork, is::Symbol...; mul::Int=1)
    return first(Iterators.drop(Iterators.filter(Base.Fix1(issetequal, is) ∘ inds, tn.indexmap[first(is)]), mul - 1))
end

"""
    in(tensor::Tensor, tn::TensorNetwork)
    in(index::Symbol, tn::TensorNetwork)

Return `true` if there is a `Tensor` in `tn` for which `==` evaluates to `true`.
This method is equivalent to `tensor ∈ tensors(tn)` code, but it's faster on large amount of tensors.
"""
Base.in(tensor::Tensor, tn::TensorNetwork) = tensor ∈ keys(tn.tensormap)
Base.in(index::Symbol, tn::TensorNetwork) = index ∈ keys(tn.indexmap)

"""
    size(tn::AbstractTensorNetwork)
    size(tn::AbstractTensorNetwork, index)

Return a mapping from indices to their dimensionalities.

If `index` is set, return the dimensionality of `index`. This is equivalent to `size(tn)[index]`.
"""
function Base.size(tn::TensorNetwork)
    return Dict{Symbol,Int}(index => size(tn, index) for index in keys(tn.indexmap))
end
Base.size(tn::TensorNetwork, index::Symbol) = size(first(tn.indexmap[index]), index)

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
    push!(tn::TensorNetwork, tensor::Tensor)

Add a new `tensor` to the Tensor Network.

See also: [`append!`](@ref), [`pop!`](@ref).
"""
function Base.push!(tn::TensorNetwork, tensor::Tensor)
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
Base.append!(tn::TensorNetwork, tensors) = (foreach(Base.Fix1(push!, tn), tensors); tn)

"""
    pop!(tn::TensorNetwork, tensor::Tensor)
    pop!(tn::TensorNetwork, i::Union{Symbol,AbstractVecOrTuple{Symbol}})

Remove a tensor from the Tensor Network and returns it. If a `Tensor` is passed, then the first tensor satisfies _egality_ (i.e. `≡` or `===`) will be removed.
If a `Symbol` or a list of `Symbol`s is passed, then remove and return the tensors that contain all the indices.

See also: [`push!`](@ref), [`delete!`](@ref).
"""
Base.pop!(tn::TensorNetwork, tensor::Tensor) = (delete!(tn, tensor); tensor)
Base.pop!(tn::TensorNetwork, i::Symbol) = pop!(tn, (i,))

function Base.pop!(tn::TensorNetwork, i::AbstractVecOrTuple{Symbol})::Vector{Tensor}
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
Base.delete!(tn::TensorNetwork, x) = (_ = pop!(tn, x); tn)

function tryprune!(tn::TensorNetwork, i::Symbol)
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

function Base.replace!(tn::TensorNetwork, old_new::Pair{Symbol,Symbol})
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
    return tn
end

function Base.replace!(tn::TensorNetwork, pair::Pair{<:Tensor,<:Tensor})
    old_tensor, new_tensor = pair

    old_tensor === new_tensor && return tn

    issetequal(inds(new_tensor), inds(old_tensor)) || throw(ArgumentError("replacing tensor indices don't match"))

    push!(tn, new_tensor)
    delete!(tn, old_tensor)

    return tn
end

function Base.replace!(tn::TensorNetwork, old_new::Pair{<:Tensor,<:TensorNetwork})
    old, new = old_new
    issetequal(inds(new; set=:open), inds(old)) || throw(ArgumentError("indices don't match"))

    # rename internal indices so there is no accidental hyperedge
    replace!(new, [index => Symbol(uuid4()) for index in filter(∈(inds(tn)), inds(new; set=:inner))])

    merge!(tn, new)
    delete!(tn, old)

    return tn
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

"""
    selectdim(tn::AbstractTensorNetwork, index::Symbol, i)

Return a copy of the [`AbstractTensorNetwork`](@ref) where `index` has been projected to dimension `i`.

See also: [`view`](@ref), [`slice!`](@ref).
"""
Base.selectdim(tn::TensorNetwork, index::Symbol, i) = @view tn[index => i]

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

"""
    groupinds!(tn::AbstractTensorNetwork, i::Symbol)

Group indices parallel to `i` and reshape the tensors accordingly.
"""
function groupinds!(tn::TensorNetwork, i)
    parinds = filter!(!=(i), inds(tn; parallelto=i))
    length(parinds) == 0 && return tn

    newtensors = map(@invoke pop!(tn, parinds ∪ (i,))) do tensor
        locᵢ = findfirst(==(i), inds(tensor))
        locs = findall(∈(parinds), inds(tensor))

        perm = collect(1:ndims(tensor))
        for (j, loc) in enumerate(locs)
            perm[loc], perm[locᵢ + j] = perm[locᵢ + j], perm[loc]
        end

        newshape = collect(size(tensor))
        newshape[locᵢ] *= prod(x -> size(tensor, x), parinds)
        deleteat!(newshape, locs)
        newinds = deleteat!(collect(inds(tensor)), locs)

        newarray = reshape(permutedims(parent(tensor), perm), newshape...)
        return Tensor(newarray, newinds)
    end

    append!(tn, newtensors)

    return tn
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

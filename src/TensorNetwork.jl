using Base: AbstractVecOrTuple
using Random
using EinExprs
using OMEinsum
using ValSplit

abstract type AbstractTensorNetwork end

"""
    TensorNetwork

Graph of interconnected tensors, representing a multilinear equation.
Graph vertices represent tensors and graph edges, tensor indices.
"""
struct TensorNetwork <: AbstractTensorNetwork
    indexmap::Dict{Symbol,Vector{Tensor}}
    tensormap::IdDict{Tensor,Vector{Symbol}}

    function TensorNetwork(tensors)
        tensormap = IdDict{Tensor,Vector{Symbol}}(tensor => inds(tensor) for tensor in tensors)

        indexmap = reduce(tensors; init = Dict{Symbol,Vector{Tensor}}()) do dict, tensor
            # TODO check for inconsistent dimensions?
            for index in inds(tensor)
                # TODO use lambda? `Tensor[]` might be reused
                push!(get!(dict, index, Tensor[]), tensor)
            end
            dict
        end

        new(indexmap, tensormap)
    end
end

TensorNetwork() = TensorNetwork(Tensor[])

"""
    copy(tn::TensorNetwork)

Return a shallow copy of a [`TensorNetwork`](@ref).
"""
Base.copy(tn::T) where {T<:AbstractTensorNetwork} = TensorNetwork(tensors(tn))

Base.summary(io::IO, tn::AbstractTensorNetwork) = print(io, "$(length(tn.tensormap))-tensors $(typeof(tn))")
Base.show(io::IO, tn::AbstractTensorNetwork) =
    print(io, "$(typeof(tn))(#tensors=$(length(tn.tensormap)), #inds=$(length(tn.indexmap)))")

"""
    tensors(tn::AbstractTensorNetwork)

Return a list of the `Tensor`s in the [`TensorNetwork`](@ref).

# Implementation details

  - As the tensors of a [`TensorNetwork`](@ref) are stored as keys of the `.tensormap` dictionary and it uses `objectid` as hash, order is not stable so it sorts for repeated evaluations.
"""
tensors(tn::AbstractTensorNetwork) = sort!(collect(keys(tn.tensormap)), by = inds)
arrays(tn::AbstractTensorNetwork) = parent.(tensors(tn))

Base.collect(tn::AbstractTensorNetwork) = tensors(tn)

"""
    inds(tn::AbstractTensorNetwork, set = :all)

Return the names of the indices in the [`TensorNetwork`](@ref).

# Keyword Arguments

  - `set`

      + `:all` (default) All indices.
      + `:open` Indices only mentioned in one tensor.
      + `:inner` Indices mentioned at least twice.
      + `:hyper` Indices mentioned at least in three tensors.
"""
Tenet.inds(tn::AbstractTensorNetwork; set::Symbol = :all, kwargs...) = inds(tn, set; kwargs...)
@valsplit 2 Tenet.inds(tn::AbstractTensorNetwork, set::Symbol, args...) = throw(MethodError(inds, "unknown set=$set"))

function Tenet.inds(tn::AbstractTensorNetwork, ::Val{:all})
    collect(keys(tn.indexmap))
end

function Tenet.inds(tn::AbstractTensorNetwork, ::Val{:open})
    map(first, Iterators.filter(((_, v),) -> length(v) == 1, tn.indexmap))
end

function Tenet.inds(tn::AbstractTensorNetwork, ::Val{:inner})
    map(first, Iterators.filter(((_, v),) -> length(v) >= 2, tn.indexmap))
end

function Tenet.inds(tn::AbstractTensorNetwork, ::Val{:hyper})
    map(first, Iterators.filter(((_, v),) -> length(v) >= 3, tn.indexmap))
end

"""
    size(tn::AbstractTensorNetwork)
    size(tn::AbstractTensorNetwork, index)

Return a mapping from indices to their dimensionalities.

If `index` is set, return the dimensionality of `index`. This is equivalent to `size(tn)[index]`.
"""
Base.size(tn::AbstractTensorNetwork) = Dict{Symbol,Int}(index => size(tn, index) for index in keys(tn.indexmap))
Base.size(tn::AbstractTensorNetwork, index::Symbol) = size(first(tn.indexmap[index]), index)

Base.eltype(tn::AbstractTensorNetwork) = promote_type(eltype.(tensors(tn))...)

"""
    push!(tn::AbstractTensorNetwork, tensor::Tensor)

Add a new `tensor` to the Tensor Network.

See also: [`append!`](@ref), [`pop!`](@ref).
"""
function Base.push!(tn::AbstractTensorNetwork, tensor::Tensor)
    tensor ∈ keys(tn.tensormap) && return tn

    # check index sizes
    for i in Iterators.filter(i -> size(tn, i) != size(tensor, i), inds(tensor) ∩ inds(tn))
        throw(DimensionMismatch("size(tensor,$i)=$(size(tensor,i)) but should be equal to size(tn,$i)=$(size(tn,i))"))
    end

    tn.tensormap[tensor] = collect(inds(tensor))
    for index in unique(inds(tensor))
        push!(get!(tn.indexmap, index, Tensor[]), tensor)
    end

    return tn
end

"""
    append!(tn::AbstractTensorNetwork, tensors::AbstractVecOrTuple{<:Tensor})

Add a list of tensors to a `TensorNetwork`.

See also: [`push!`](@ref), [`merge!`](@ref).
"""
Base.append!(tn::AbstractTensorNetwork, tensors) = (foreach(Base.Fix1(push!, tn), tensors); tn)

"""
    merge!(self::AbstractTensorNetwork, others::AbstractTensorNetwork...)
    merge(self::AbstractTensorNetwork, others::AbstractTensorNetwork...)

Fuse various [`TensorNetwork`](@ref)s into one.

See also: [`append!`](@ref).
"""
Base.merge!(self::AbstractTensorNetwork, other::AbstractTensorNetwork) = append!(self, tensors(other))
Base.merge!(self::AbstractTensorNetwork, others::AbstractTensorNetwork...) = foldl(merge!, others; init = self)
Base.merge(self::AbstractTensorNetwork, others::AbstractTensorNetwork...) = merge!(copy(self), others...)

"""
    pop!(tn::AbstractTensorNetwork, tensor::Tensor)
    pop!(tn::AbstractTensorNetwork, i::Union{Symbol,AbstractVecOrTuple{Symbol}})

Remove a tensor from the Tensor Network and returns it. If a `Tensor` is passed, then the first tensor satisfies _egality_ (i.e. `≡` or `===`) will be removed.
If a `Symbol` or a list of `Symbol`s is passed, then remove and return the tensors that contain all the indices.

See also: [`push!`](@ref), [`delete!`](@ref).
"""
Base.pop!(tn::AbstractTensorNetwork, tensor::Tensor) = (delete!(tn, tensor); tensor)
Base.pop!(tn::AbstractTensorNetwork, i::Symbol) = pop!(tn, (i,))

function Base.pop!(tn::AbstractTensorNetwork, i::AbstractVecOrTuple{Symbol})::Vector{Tensor}
    tensors = select(tn, i)
    for tensor in tensors
        _ = pop!(tn, tensor)
    end

    return tensors
end

"""
    delete!(tn::AbstractTensorNetwork, x)

Like [`pop!`](@ref) but return the [`TensorNetwork`](@ref) instead.
"""
Base.delete!(tn::AbstractTensorNetwork, x) = (_ = pop!(tn, x); tn)

tryprune!(tn::AbstractTensorNetwork, i::Symbol) = (x = isempty(tn.indexmap[i]) && delete!(tn.indexmap, i); x)

function Base.delete!(tn::AbstractTensorNetwork, tensor::Tensor)
    for index in unique(inds(tensor))
        filter!(Base.Fix1(!==, tensor), tn.indexmap[index])
        tryprune!(tn, index)
    end
    delete!(tn.tensormap, tensor)

    return tn
end

"""
    replace!(tn::AbstractTensorNetwork, old => new...)
    replace(tn::AbstractTensorNetwork, old => new...)

Replace the element in `old` with the one in `new`. Depending on the types of `old` and `new`, the following behaviour is expected:

  - If `Symbol`s, it will correspond to a index renaming.
  - If `Tensor`s, first element that satisfies _egality_ (`≡` or `===`) will be replaced.
"""
Base.replace!(tn::AbstractTensorNetwork, old_new::Pair...) = replace!(tn, old_new)
function Base.replace!(tn::AbstractTensorNetwork, old_new::Base.AbstractVecOrTuple{Pair})
    for pair in old_new
        replace!(tn, pair)
    end
    return tn
end
Base.replace(tn::AbstractTensorNetwork, old_new::Pair...) = replace(tn, old_new)
Base.replace(tn::AbstractTensorNetwork, old_new) = replace!(copy(tn), old_new)

function Base.replace!(tn::AbstractTensorNetwork, pair::Pair{<:Tensor,<:Tensor})
    old_tensor, new_tensor = pair
    issetequal(inds(new_tensor), inds(old_tensor)) || throw(ArgumentError("replacing tensor indices don't match"))

    push!(tn, new_tensor)
    delete!(tn, old_tensor)

    return tn
end

function Base.replace!(tn::AbstractTensorNetwork, old_new::Pair{Symbol,Symbol}...)
    first.(old_new) ⊆ keys(tn.indexmap) ||
        throw(ArgumentError("set of old indices must be a subset of current indices"))
    isdisjoint(last.(old_new), keys(tn.indexmap)) ||
        throw(ArgumentError("set of new indices must be disjoint to current indices"))
    for pair in old_new
        replace!(tn, pair)
    end
    return tn
end

function Base.replace!(tn::AbstractTensorNetwork, old_new::Pair{Symbol,Symbol})
    old, new = old_new
    old ∈ keys(tn.indexmap) || throw(ArgumentError("index $old does not exist"))
    new ∉ keys(tn.indexmap) || throw(ArgumentError("index $new is already present"))

    # NOTE `copy` because collection underneath is mutated
    for tensor in copy(tn.indexmap[old])
        # NOTE do not `delete!` before `push!` as indices can be lost due to `tryprune!`
        push!(tn, replace(tensor, old_new))
        delete!(tn, tensor)
    end

    delete!(tn.indexmap, old)

    return tn
end

function Base.replace!(tn::AbstractTensorNetwork, old_new::Pair{<:Tensor,<:AbstractTensorNetwork})
    old, new = old_new
    issetequal(inds(new, set = :open), inds(old)) || throw(ArgumentError("indices don't match match"))

    # rename internal indices so there is no accidental hyperedge
    replace!(new, [index => Symbol(uuid4()) for index in filter(∈(inds(tn)), inds(new, set = :inner))]...)

    merge!(tn, new)
    delete!(tn, old)

    return tn
end

"""
    select(tn::AbstractTensorNetwork, i)

Return tensors whose indices match with the list of indices `i`.
"""
select(tn::AbstractTensorNetwork, i::Symbol) = copy(tn.indexmap[i])
select(tn::AbstractTensorNetwork, is::AbstractVecOrTuple{Symbol}) = select(⊆, tn, is)

function select(selector, tn::TensorNetwork, is::AbstractVecOrTuple{Symbol})
    filter(Base.Fix1(selector, is) ∘ inds, tn.indexmap[first(is)])
end

function Base.getindex(tn::TensorNetwork, is::Symbol...; mul::Int = 1)
    first(Iterators.drop(Iterators.filter(Base.Fix1(issetequal, is) ∘ inds, tn.indexmap[first(is)]), mul - 1))
end

"""
    in(tensor::Tensor, tn::AbstractTensorNetwork)
    in(index::Symbol, tn::AbstractTensorNetwork)

Return `true` if there is a `Tensor` in `tn` for which `==` evaluates to `true`.
This method is equivalent to `tensor ∈ tensors(tn)` code, but it's faster on large amount of tensors.
"""
Base.in(tensor::Tensor, tn::AbstractTensorNetwork) = tensor ∈ keys(tn.tensormap)
Base.in(index::Symbol, tn::AbstractTensorNetwork) = index ∈ keys(tn.indexmap)

"""
    slice!(tn::AbstractTensorNetwork, index::Symbol, i)

In-place projection of `index` on dimension `i`.

See also: [`selectdim`](@ref), [`view`](@ref).
"""
function slice!(tn::AbstractTensorNetwork, label::Symbol, i)
    for tensor in pop!(tn, label)
        push!(tn, selectdim(tensor, label, i))
    end

    return tn
end

"""
    selectdim(tn::AbstractTensorNetwork, index::Symbol, i)

Return a copy of the [`TensorNetwork`](@ref) where `index` has been projected to dimension `i`.

See also: [`view`](@ref), [`slice!`](@ref).
"""
Base.selectdim(tn::AbstractTensorNetwork, label::Symbol, i) = @view tn[label=>i]

"""
    view(tn::AbstractTensorNetwork, index => i...)

Return a copy of the [`TensorNetwork`](@ref) where each `index` has been projected to dimension `i`.
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
    ::Type{TensorNetwork},
    n::Integer,
    regularity::Integer;
    out = 0,
    dim = 2:9,
    seed = nothing,
    globalind = false,
)
    !isnothing(seed) && Random.seed!(seed)

    inds = letter.(randperm(n * regularity ÷ 2 + out))
    size_dict = Dict(ind => rand(dim) for ind in inds)

    outer_inds = Iterators.take(inds, out) |> collect
    inner_inds = Iterators.drop(inds, out) |> collect

    candidate_inds =
        [outer_inds, Iterators.flatten(Iterators.repeated(inner_inds, 2))] |> Iterators.flatten |> collect |> shuffle

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

    tensors = Tensor[Tensor(rand([size_dict[ind] for ind in input]...), tuple(input...)) for input in inputs]
    TensorNetwork(tensors)
end

"""
    einexpr(tn::AbstractTensorNetwork; optimizer = EinExprs.Greedy, output = inds(tn, :open), kwargs...)

Search a contraction path for the given [`TensorNetwork`](@ref) and return it as a `EinExpr`.

# Keyword Arguments

  - `optimizer` Contraction path optimizer. Check [`EinExprs`](https://github.com/bsc-quantic/EinExprs.jl) documentation for more info.
  - `outputs` Indices that won't be contracted. Defaults to open indices.
  - `kwargs` Options to be passed to the optimizer.

See also: [`contract`](@ref).
"""
EinExprs.einexpr(tn::AbstractTensorNetwork; optimizer = Greedy, outputs = inds(tn, :open), kwargs...) = einexpr(
    optimizer,
    EinExpr(
        outputs,
        [EinExpr(inds(tensor), Dict(index => size(tensor, index) for index in inds(tensor))) for tensor in tensors(tn)],
    );
    kwargs...,
)

# TODO sequence of indices?
# TODO what if parallel neighbour indices?
"""
    contract!(tn::AbstractTensorNetwork, index)

In-place contraction of tensors connected to `index`.

See also: [`contract`](@ref).
"""
function contract!(tn::AbstractTensorNetwork, i)
    tensor = reduce(pop!(tn, i)) do acc, tensor
        contract(acc, tensor, dims = i)
    end

    push!(tn, tensor)
    return tn
end

"""
    contract(tn::AbstractTensorNetwork; kwargs...)

Contract a [`TensorNetwork`](@ref). The contraction order will be first computed by [`einexpr`](@ref).

The `kwargs` will be passed down to the [`einexpr`](@ref) function.

See also: [`einexpr`](@ref), [`contract!`](@ref).
"""
function contract(tn::AbstractTensorNetwork; path = einexpr(tn))
    # TODO does `first` work always?
    length(path.args) == 0 && return select(tn, inds(path)) |> first

    intermediates = map(subpath -> contract(tn; path = subpath), path.args)
    contract(intermediates...; dims = suminds(path))
end

contract!(t::Tensor, tn::AbstractTensorNetwork; kwargs...) = contract!(tn, t; kwargs...)
contract!(tn::AbstractTensorNetwork, t::Tensor; kwargs...) = (push!(tn, t); contract(tn; kwargs...))
contract(t::Tensor, tn::AbstractTensorNetwork; kwargs...) = contract(tn, t; kwargs...)
contract(tn::AbstractTensorNetwork, t::Tensor; kwargs...) = contract!(copy(tn), t; kwargs...)

struct TNSampler{T<:AbstractTensorNetwork} <: Random.Sampler{T}
    config::Dict{Symbol,Any}

    TNSampler{T}(; kwargs...) where {T} = new{T}(kwargs)
end

Base.eltype(::TNSampler{T}) where {T} = T

Base.getproperty(obj::TNSampler, name::Symbol) = name === :config ? getfield(obj, :config) : obj.config[name]
Base.get(obj::TNSampler, name, default) = get(obj.config, name, default)

Base.rand(T::Type{<:AbstractTensorNetwork}; kwargs...) = rand(Random.default_rng(), T; kwargs...)
Base.rand(rng::AbstractRNG, T::Type{<:AbstractTensorNetwork}; kwargs...) = rand(rng, TNSampler{T}(; kwargs...))

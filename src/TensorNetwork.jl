using Base: AbstractVecOrTuple
using Random
using EinExprs
using OMEinsum
using ValSplit
using LinearAlgebra

"""
    TensorNetwork

Graph of interconnected tensors, representing a multilinear equation.
Graph vertices represent tensors and graph edges, tensor indices.
"""
struct TensorNetwork
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
Base.copy(tn::TensorNetwork) = TensorNetwork(tensors(tn))

Base.similar(tn::TensorNetwork) = TensorNetwork(similar.(tensors(tn)))
Base.zero(tn::TensorNetwork) = TensorNetwork(zero.(tensors(tn)))

Base.summary(io::IO, tn::TensorNetwork) = print(io, "$(length(tn.tensormap))-tensors TensorNetwork")
Base.show(io::IO, tn::TensorNetwork) =
    print(io, "TensorNetwork (#tensors=$(length(tn.tensormap)), #inds=$(length(tn.indexmap)))")

function Base.:(==)(a::TensorNetwork, b::TensorNetwork)
    issetequal(inds(a), inds(b)) || return false
    all(tensors(a)) do ta
        tb = b[inds(ta)...]
        ta == tb
    end
end

function Base.isapprox(a::TensorNetwork, b::TensorNetwork; kwargs...)
    issetequal(inds(a), inds(b)) || return false
    all(tensors(a)) do ta
        tb = b[inds(ta)...]
        isapprox(ta, tb; kwargs...)
    end
end

"""
    tensors(tn::TensorNetwork)

Return a list of the `Tensor`s in the [`TensorNetwork`](@ref).

# Implementation details

  - As the tensors of a [`TensorNetwork`](@ref) are stored as keys of the `.tensormap` dictionary and it uses `objectid` as hash, order is not stable so it sorts for repeated evaluations.
"""
tensors(tn::TensorNetwork) = sort!(collect(keys(tn.tensormap)), by = inds)
arrays(tn::TensorNetwork) = parent.(tensors(tn))

Base.collect(tn::TensorNetwork) = tensors(tn)

"""
    inds(tn::TensorNetwork, set = :all)

Return the names of the indices in the [`TensorNetwork`](@ref).

# Keyword Arguments

  - `set`

      + `:all` (default) All indices.
      + `:open` Indices only mentioned in one tensor.
      + `:inner` Indices mentioned at least twice.
      + `:hyper` Indices mentioned at least in three tensors.
"""
Tenet.inds(tn::TensorNetwork; set::Symbol = :all, kwargs...) = inds(tn, set; kwargs...)
@valsplit 2 Tenet.inds(tn::TensorNetwork, set::Symbol, args...) = throw(MethodError(inds, "unknown set=$set"))

function Tenet.inds(tn::TensorNetwork, ::Val{:all})
    collect(keys(tn.indexmap))
end

function Tenet.inds(tn::TensorNetwork, ::Val{:open})
    map(first, Iterators.filter(((_, v),) -> length(v) == 1, tn.indexmap))
end

function Tenet.inds(tn::TensorNetwork, ::Val{:inner})
    map(first, Iterators.filter(((_, v),) -> length(v) >= 2, tn.indexmap))
end

function Tenet.inds(tn::TensorNetwork, ::Val{:hyper})
    map(first, Iterators.filter(((_, v),) -> length(v) >= 3, tn.indexmap))
end

"""
    size(tn::TensorNetwork)
    size(tn::TensorNetwork, index)

Return a mapping from indices to their dimensionalities.

If `index` is set, return the dimensionality of `index`. This is equivalent to `size(tn)[index]`.
"""
Base.size(tn::TensorNetwork) = Dict{Symbol,Int}(index => size(tn, index) for index in keys(tn.indexmap))
Base.size(tn::TensorNetwork, index::Symbol) = size(first(tn.indexmap[index]), index)

Base.eltype(tn::TensorNetwork) = promote_type(eltype.(tensors(tn))...)

"""
    select(tn::TensorNetwork, :containing, i)
    select(tn::TensorNetwork, :any, i)
    select(tn::TensorNetwork, :all, i)

Return tensors whose indices match with the list of indices `i`.
"""
@valsplit 2 function select(tn::TensorNetwork, query::Symbol, args...)
    error("Query ':$query' not defined for TensorNetwork")
end

function select(selector, tn::TensorNetwork, is::AbstractVecOrTuple{Symbol})
    filter(Base.Fix1(selector, is) ∘ inds, tn.indexmap[first(is)])
end

# TODO better naming?
select(tn::TensorNetwork, ::Val{:containing}, i::Symbol) = copy(tn.indexmap[i])
select(tn::TensorNetwork, ::Val{:containing}, is::AbstractVecOrTuple{Symbol}) = select(⊆, tn, is)

select(tn::TensorNetwork, ::Val{:any}, i::Symbol) = select(!isdisjoint, tn, [i])
select(tn::TensorNetwork, ::Val{:any}, is::AbstractVecOrTuple{Symbol}) = select(!isdisjoint, tn, is)

# TODO `:all` is an alias for `:containing`. maybe change?
select(tn::TensorNetwork, ::Val{:all}, i::Symbol) = copy(tn.indexmap[i])
select(tn::TensorNetwork, ::Val{:all}, is::AbstractVecOrTuple{Symbol}) = select(⊆, tn, is)

function Base.getindex(tn::TensorNetwork, is::Symbol...; mul::Int = 1)
    first(Iterators.drop(Iterators.filter(Base.Fix1(issetequal, is) ∘ inds, tn.indexmap[first(is)]), mul - 1))
end

function neighbors(tn::TensorNetwork, tensor::Tensor; open::Bool = true)
    @assert tensor ∈ tn "Tensor not found in TensorNetwork"
    tensors = mapreduce(∪, inds(tensor)) do index
        select(tn, :any, index)
    end
    open && filter!(x -> x !== tensor, tensors)
    tensors
end

function neighbors(tn::TensorNetwork, i::Symbol; open::Bool = true)
    @assert i ∈ tn "Index $i not found in TensorNetwork"
    tensors = mapreduce(inds, ∪, select(tn, :any, i))
    # open && filter!(x -> x !== i, tensors)
    tensors
end

"""
    push!(tn::TensorNetwork, tensor::Tensor)

Add a new `tensor` to the Tensor Network.

See also: [`append!`](@ref), [`pop!`](@ref).
"""
function Base.push!(tn::TensorNetwork, tensor::Tensor)
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
    append!(tn::TensorNetwork, tensors::AbstractVecOrTuple{<:Tensor})

Add a list of tensors to a `TensorNetwork`.

See also: [`push!`](@ref), [`merge!`](@ref).
"""
Base.append!(tn::TensorNetwork, tensors) = (foreach(Base.Fix1(push!, tn), tensors); tn)

"""
    merge!(self::TensorNetwork, others::TensorNetwork...)
    merge(self::TensorNetwork, others::TensorNetwork...)

Fuse various [`TensorNetwork`](@ref)s into one.

See also: [`append!`](@ref).
"""
Base.merge!(self::TensorNetwork, other::TensorNetwork) = append!(self, tensors(other))
Base.merge!(self::TensorNetwork, others::TensorNetwork...) = foldl(merge!, others; init = self)
Base.merge(self::TensorNetwork, others::TensorNetwork...) = merge!(copy(self), others...)

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
    tensors = select(tn, :any, i)
    for tensor in tensors
        _ = pop!(tn, tensor)
    end

    return tensors
end

"""
    delete!(tn::TensorNetwork, x)

Like [`pop!`](@ref) but return the [`TensorNetwork`](@ref) instead.
"""
Base.delete!(tn::TensorNetwork, x) = (_ = pop!(tn, x); tn)

tryprune!(tn::TensorNetwork, i::Symbol) = (x = isempty(tn.indexmap[i]) && delete!(tn.indexmap, i); x)

function Base.delete!(tn::TensorNetwork, tensor::Tensor)
    for index in unique(inds(tensor))
        filter!(Base.Fix1(!==, tensor), tn.indexmap[index])
        tryprune!(tn, index)
    end
    delete!(tn.tensormap, tensor)

    return tn
end

"""
    replace!(tn::TensorNetwork, old => new...)
    replace(tn::TensorNetwork, old => new...)

Replace the element in `old` with the one in `new`. Depending on the types of `old` and `new`, the following behaviour is expected:

  - If `Symbol`s, it will correspond to a index renaming.
  - If `Tensor`s, first element that satisfies _egality_ (`≡` or `===`) will be replaced.
"""
Base.replace!(tn::TensorNetwork, old_new::Pair...) = replace!(tn, old_new)
function Base.replace!(tn::TensorNetwork, old_new::Base.AbstractVecOrTuple{Pair})
    for pair in old_new
        replace!(tn, pair)
    end
    return tn
end
Base.replace(tn::TensorNetwork, old_new::Pair...) = replace(tn, old_new)
Base.replace(tn::TensorNetwork, old_new) = replace!(copy(tn), old_new)

function Base.replace!(tn::TensorNetwork, pair::Pair{<:Tensor,<:Tensor})
    old_tensor, new_tensor = pair
    issetequal(inds(new_tensor), inds(old_tensor)) || throw(ArgumentError("replacing tensor indices don't match"))

    push!(tn, new_tensor)
    delete!(tn, old_tensor)

    return tn
end

function Base.replace!(tn::TensorNetwork, old_new::Pair{Symbol,Symbol}...)
    first.(old_new) ⊆ keys(tn.indexmap) ||
        throw(ArgumentError("set of old indices must be a subset of current indices"))
    isdisjoint(last.(old_new), keys(tn.indexmap)) ||
        throw(ArgumentError("set of new indices must be disjoint to current indices"))
    for pair in old_new
        replace!(tn, pair)
    end
    return tn
end

function Base.replace!(tn::TensorNetwork, old_new::Pair{Symbol,Symbol})
    old, new = old_new
    old ∈ keys(tn.indexmap) || throw(ArgumentError("index $old does not exist"))
    old == new && return tn
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

function Base.replace!(tn::TensorNetwork, old_new::Pair{<:Tensor,<:TensorNetwork})
    old, new = old_new
    issetequal(inds(new, set = :open), inds(old)) || throw(ArgumentError("indices don't match match"))

    # rename internal indices so there is no accidental hyperedge
    replace!(new, [index => Symbol(uuid4()) for index in filter(∈(inds(tn)), inds(new, set = :inner))]...)

    merge!(tn, new)
    delete!(tn, old)

    return tn
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
    slice!(tn::TensorNetwork, index::Symbol, i)

In-place projection of `index` on dimension `i`.

See also: [`selectdim`](@ref), [`view`](@ref).
"""
function slice!(tn::TensorNetwork, label::Symbol, i)
    for tensor in pop!(tn, label)
        push!(tn, selectdim(tensor, label, i))
    end

    return tn
end

"""
    selectdim(tn::TensorNetwork, index::Symbol, i)

Return a copy of the [`TensorNetwork`](@ref) where `index` has been projected to dimension `i`.

See also: [`view`](@ref), [`slice!`](@ref).
"""
Base.selectdim(tn::TensorNetwork, label::Symbol, i) = @view tn[label=>i]

"""
    view(tn::TensorNetwork, index => i...)

Return a copy of the [`TensorNetwork`](@ref) where each `index` has been projected to dimension `i`.
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

struct TNSampler <: Random.Sampler{TensorNetwork}
    config::Dict{Symbol,Any}

    TNSampler(; kwargs...) = new(kwargs)
end

Base.eltype(::TNSampler) = TensorNetwork

Base.getproperty(obj::TNSampler, name::Symbol) = name === :config ? getfield(obj, :config) : obj.config[name]
Base.get(obj::TNSampler, name, default) = get(obj.config, name, default)

Base.rand(::Type{TensorNetwork}; kwargs...) = rand(Random.default_rng(), TensorNetwork; kwargs...)
Base.rand(rng::AbstractRNG, ::Type{TensorNetwork}; kwargs...) = rand(rng, TNSampler(; kwargs...))

"""
    einexpr(tn::TensorNetwork; optimizer = EinExprs.Greedy, output = inds(tn, :open), kwargs...)

Search a contraction path for the given [`TensorNetwork`](@ref) and return it as a `EinExpr`.

# Keyword Arguments

  - `optimizer` Contraction path optimizer. Check [`EinExprs`](https://github.com/bsc-quantic/EinExprs.jl) documentation for more info.
  - `outputs` Indices that won't be contracted. Defaults to open indices.
  - `kwargs` Options to be passed to the optimizer.

See also: [`contract`](@ref).
"""
EinExprs.einexpr(tn::TensorNetwork; optimizer = Greedy, outputs = inds(tn, :open), kwargs...) = einexpr(
    optimizer,
    sum(
        [EinExpr(inds(tensor), Dict(index => size(tensor, index) for index in inds(tensor))) for tensor in tensors(tn)],
        skip = outputs,
    );
    kwargs...,
)

function Base.conj!(tn::TensorNetwork)
    foreach(conj!, tensors(tn))
    return tn
end

Base.conj(tn::TensorNetwork) = TensorNetwork(map(conj, tensors(tn)))

# TODO sequence of indices?
# TODO what if parallel neighbour indices?
"""
    contract!(tn::TensorNetwork, index)

In-place contraction of tensors connected to `index`.

See also: [`contract`](@ref).
"""
function contract!(tn::TensorNetwork, i)
    _tensors = sort!(select(tn, :any, i), by = length)
    tensor = contract(TensorNetwork(_tensors))
    delete!(tn, i)
    push!(tn, tensor)
    return tn
end
function contract(tn::TensorNetwork, i)
    tn = copy(tn)
    contract!(tn, i)
end
contract!(tn::TensorNetwork, i::Symbol) = contract!(tn, [i])
contract(tn::TensorNetwork, i::Symbol) = contract(tn, [i])

"""
    contract(tn::TensorNetwork; kwargs...)

Contract a [`TensorNetwork`](@ref). The contraction order will be first computed by [`einexpr`](@ref).

The `kwargs` will be passed down to the [`einexpr`](@ref) function.

See also: [`einexpr`](@ref), [`contract!`](@ref).
"""
function contract(tn::TensorNetwork; path = einexpr(tn))
    length(path.args) == 0 && return tn[inds(path)...]

    intermediates = map(subpath -> contract(tn; path = subpath), path.args)
    contract(intermediates...; dims = suminds(path))
end

contract!(t::Tensor, tn::TensorNetwork; kwargs...) = contract!(tn, t; kwargs...)
contract!(tn::TensorNetwork, t::Tensor; kwargs...) = (push!(tn, t); contract(tn; kwargs...))
contract(t::Tensor, tn::TensorNetwork; kwargs...) = contract(tn, t; kwargs...)
contract(tn::TensorNetwork, t::Tensor; kwargs...) = contract!(copy(tn), t; kwargs...)

function LinearAlgebra.svd!(tn::TensorNetwork; left_inds = Symbol[], right_inds = Symbol[], kwargs...)
    tensor = tn[left_inds ∪ right_inds...]
    U, s, Vt = svd(tensor; left_inds, right_inds, kwargs...)
    replace!(tn, tensor => TensorNetwork([U, s, Vt]))
    return tn
end

function LinearAlgebra.qr!(tn::TensorNetwork; left_inds = Symbol[], right_inds = Symbol[], kwargs...)
    tensor = tn[left_inds ∪ right_inds...]
    Q, R = qr(tensor; left_inds, right_inds, kwargs...)
    replace!(tn, tensor => TensorNetwork([Q, R]))
    return tn
end

function LinearAlgebra.lu!(tn::TensorNetwork; left_inds = Symbol[], right_inds = Symbol[], kwargs...)
    tensor = tn[left_inds ∪ right_inds...]
    L, U = lu(tensor; left_inds, right_inds, kwargs...)
    replace!(tn, tensor => TensorNetwork([L, U]))
    return tn
end

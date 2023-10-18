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
    incidence::IncidenceMatrix{Int}
    indexmap::IndexBijection
    tensormap::TensorBijection
end

TensorNetwork() = TensorNetwork(IncidenceMatrix{Int}(), IndexBijection(), TensorBijection())
function TensorNetwork(tensors)
    indices = unique(Iterators.flatmap(inds, tensors))
    indexmap = IndexBijection()
    for (j, index) in enumerate(indices)
        indexmap[j] = index
    end

    tensormap = TensorBijection()
    for (i, tensor) in enumerate(tensors)
        tensormap[i] = tensor
    end

    incidence = IncidenceMatrix{Int}()
    for (i, tensor) in enumerate(tensors), j in Iterators.map(indexmap, inds(tensor))
        incidence[i, j] = true
    end

    # check for inconsistent dimensions
    for (j, index) in indexmap
        is = incidence[:, j]
        allequal(Iterators.map(i -> size(tensormap[i], index), is)) ||
            throw(DimensionMismatch("Different sizes specified for index $index"))
    end

    return TensorNetwork(incidence, indexmap, tensormap)
end

"""
    copy(tn::TensorNetwork)

Return a shallow copy of a [`TensorNetwork`](@ref).
"""
Base.copy(tn::T) where {T<:AbstractTensorNetwork} = T(map(fieldnames(T)) do field
    (field === :indices ? deepcopy : copy)(getfield(tn, field))
end...)

Base.summary(io::IO, x::AbstractTensorNetwork) = print(io, "$(length(x))-tensors $(typeof(x))")
Base.show(io::IO, tn::AbstractTensorNetwork) =
    print(io, "$(typeof(tn))(#tensors=$(length(tn.tensors)), #inds=$(length(tn.indices)))")

"""
    tensors(tn::AbstractTensorNetwork)

Return a list of the `Tensor`s in the [`TensorNetwork`](@ref).
"""
tensors(tn::AbstractTensorNetwork) = tn.tensors
arrays(tn::AbstractTensorNetwork) = parent.(tensors(tn))

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
inds(tn::AbstractTensorNetwork; set::Symbol = :all, kwargs...) = inds(tn, set; kwargs...)
@valsplit 2 inds(tn::AbstractTensorNetwork, set::Symbol, args...) = throw(MethodError(inds, "unknown set=$set"))
inds(tn::AbstractTensorNetwork, ::Val{:all}) = collect(keys(tn.indices))
inds(tn::AbstractTensorNetwork, ::Val{:open}) = map(first, Iterators.filter(==(1) ∘ length ∘ last, tn.indices))
inds(tn::AbstractTensorNetwork, ::Val{:inner}) = map(first, Iterators.filter(>=(2) ∘ length ∘ last, tn.indices))
inds(tn::AbstractTensorNetwork, ::Val{:hyper}) = map(first, Iterators.filter(>=(3) ∘ length ∘ last, tn.indices))

"""
    size(tn::AbstractTensorNetwork)
    size(tn::AbstractTensorNetwork, index)

Return a mapping from indices to their dimensionalities.

If `index` is set, return the dimensionality of `index`. This is equivalent to `size(tn)[index]`.
"""
Base.size(tn::AbstractTensorNetwork) = Dict(i => size(tn, i) for (i, x) in tn.indices)
Base.size(tn::AbstractTensorNetwork, i::Symbol) = size(tn.tensors[first(tn.indices[i])], i)

Base.eltype(tn::AbstractTensorNetwork) = promote_type(eltype.(tensors(tn))...)

"""
    push!(tn::AbstractTensorNetwork, tensor::Tensor)

Add a new `tensor` to the Tensor Network.

See also: [`append!`](@ref), [`pop!`](@ref).
"""
function Base.push!(tn::AbstractTensorNetwork, tensor::Tensor)
    for i in Iterators.filter(i -> size(tn, i) != size(tensor, i), inds(tensor) ∩ inds(tn))
        throw(DimensionMismatch("size(tensor,$i)=$(size(tensor,i)) but should be equal to size(tn,$i)=$(size(tn,i))"))
    end

    push!(tn.tensors, tensor)

    for i in inds(tensor)
        push!(get!(tn.indices, i, Int[]), length(tn.tensors))
    end

    return tn
end

"""
    append!(tn::AbstractTensorNetwork, tensors::AbstractVecOrTuple{<:Tensor})

Add a list of tensors to a `TensorNetwork`.

See also: [`push!`](@ref), [`merge!`](@ref).
"""
function Base.append!(tn::AbstractTensorNetwork, ts::AbstractVecOrTuple{<:Tensor})
    for tensor in ts
        push!(tn, tensor)
    end
    tn
end

"""
    merge!(self::AbstractTensorNetwork, others::AbstractTensorNetwork...)
    merge(self::AbstractTensorNetwork, others::AbstractTensorNetwork...)

Fuse various [`TensorNetwork`](@ref)s into one.

See also: [`append!`](@ref).
"""
Base.merge!(self::AbstractTensorNetwork, other::AbstractTensorNetwork) = append!(self, tensors(other))
Base.merge!(self::AbstractTensorNetwork, others::AbstractTensorNetwork...) = foldl(merge!, others; init = self)
Base.merge(self::AbstractTensorNetwork, others::AbstractTensorNetwork...) = merge!(copy(self), others...)

function Base.popat!(tn::AbstractTensorNetwork, i::Integer)
    tensor = popat!(tn.tensors, i)

    # unlink indices
    for index in unique(inds(tensor))
        filter!(!=(i), tn.indices[index])
        isempty(tn.indices[index]) && delete!(tn.indices, index)
    end

    # update tensor positions in `tn.indices`
    for locations in values(tn.indices)
        map!(locations, locations) do loc
            loc > i ? loc - 1 : loc
        end
    end

    return tensor
end

"""
    pop!(tn::AbstractTensorNetwork, tensor::Tensor)
    pop!(tn::AbstractTensorNetwork, i::Union{Symbol,AbstractVecOrTuple{Symbol}})

Remove a tensor from the Tensor Network and returns it. If a `Tensor` is passed, then the first tensor satisfies _egality_ (i.e. `≡` or `===`) will be removed.
If a `Symbol` or a list of `Symbol`s is passed, then remove and return the tensors that contain all the indices.

See also: [`push!`](@ref), [`delete!`](@ref).
"""
function Base.pop!(tn::AbstractTensorNetwork, tensor::Tensor)
    i = findfirst(t -> t === tensor, tn.tensors)
    popat!(tn, i)
end

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

    # check if old and new tensors are compatible
    if !issetequal(inds(new_tensor), inds(old_tensor))
        throw(ArgumentError("New tensor indices do not match the existing tensor inds"))
    end

    # replace existing `Tensor` with new `Tensor`
    i = findfirst(t -> t === old_tensor, tn.tensors)
    splice!(tn.tensors, i, [new_tensor])

    return tn
end

function Base.replace!(tn::AbstractTensorNetwork, old_new::Pair{Symbol,Symbol})
    old, new = old_new
    new ∈ inds(tn) && throw(ArgumentError("new symbol $new is already present"))

    push!(tn.indices, new => pop!(tn.indices, old))

    for i in tn.indices[new]
        tn.tensors[i] = replace(tn.tensors[i], old_new)
    end

    return tn
end

function Base.replace!(tn::AbstractTensorNetwork, old_new::Pair{<:Tensor,<:AbstractTensorNetwork})
    old, new = old_new
    issetequal(inds(new, set = :open), inds(old)) || throw(ArgumentError("indices must match"))

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
select(tn::AbstractTensorNetwork, i::AbstractVecOrTuple{Symbol}) = filter(Base.Fix1(⊆, i) ∘ inds, tensors(tn))
select(tn::AbstractTensorNetwork, i::Symbol) = map(x -> tn.tensors[x], unique(tn.indices[i]))

"""
    in(tensor::Tensor, tn::AbstractTensorNetwork)

Return `true` if there is a `Tensor` in `tn` for which `==` evaluates to `true`.
This method is equivalent to `tensor ∈ tensors(tn)` code, but it's faster on large amount of tensors.
"""
Base.in(tensor::Tensor, tn::AbstractTensorNetwork) = in(tensor, select(tn, inds(tensor)))

"""
    slice!(tn::AbstractTensorNetwork, index::Symbol, i)

In-place projection of `index` on dimension `i`.

See also: [`selectdim`](@ref), [`view`](@ref).
"""
function slice!(tn::AbstractTensorNetwork, label::Symbol, i)
    for tensor in select(tn, label)
        pos = findfirst(t -> t === tensor, tn.tensors)
        tn.tensors[pos] = selectdim(tensor, label, i)
    end

    i isa Integer && delete!(tn.indices, label)

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
function Base.view(tn::AbstractTensorNetwork, slices::Pair{Symbol,<:Any}...)
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

using Base: AbstractVecOrTuple
using Random
using EinExprs
using OMEinsum
using ValSplit

"""
    Ansatz

Type representing the predefined form of the Tensor Network.
"""
abstract type Ansatz end

"""
    Arbitrary

Tensor Networks without a predefined form.
"""
abstract type Arbitrary <: Ansatz end

# NOTE currently, these are implementation details
function checkmeta end
function metadata end

"""
    TensorNetwork{Ansatz}

Graph of interconnected tensors, representing a multilinear equation.
Graph vertices represent tensors and graph edges, tensor indices.
"""
struct TensorNetwork{A<:Ansatz,M<:NamedTuple}
    indices::Dict{Symbol,Vector{Int}}
    tensors::Vector{Tensor}
    metadata::M

    function TensorNetwork{A}(tensors; metadata...) where {A}
        indices = reduce(enumerate(tensors); init = Dict{Symbol,Vector{Int}}([])) do dict, (i, tensor)
            mergewith(vcat, dict, Dict([index => [i] for index in inds(tensor)]))
        end

        # Check for inconsistent dimensions
        for (index, idxs) in indices
            allequal(Iterators.map(i -> size(tensors[i], index), idxs)) ||
                throw(DimensionMismatch("Different sizes specified for index $index"))
        end

        M = Tenet.metadata(A)
        metadata = M((; metadata...))

        tn = new{A,M}(indices, tensors, metadata)

        checkansatz(tn)
        return tn
    end
end

TensorNetwork{A}(; metadata...) where {A<:Ansatz} = TensorNetwork{A}(Tensor[]; metadata...)

# ansatz defaults to `Arbitrary`
TensorNetwork(args...; kwargs...) = TensorNetwork{Arbitrary}(args...; kwargs...)

# TODO maybe rename it as `convert` method?
TensorNetwork{A}(tn::TensorNetwork{B}; metadata...) where {A,B} =
    TensorNetwork{A}(tensors(tn); merge(tn.metadata, metadata)...)

# TODO do sth to skip checkansatz? like @inbounds
function checkansatz(tn::TensorNetwork{A}) where {A<:Ansatz}
    for T in superansatzes(A)
        checkmeta(T, tn) || throw(ErrorException("\"$T\" metadata is not valid"))
    end
end

checkmeta(::Type{<:Ansatz}, ::TensorNetwork) = true
checkmeta(tn::TensorNetwork{T}) where {T<:Ansatz} = all(A -> checkmeta(A, tn), superansatzes(T))

metadata(::Type{<:Ansatz}) = NamedTuple{(),Tuple{}}
metadata(T::Type{<:Arbitrary}) = metadata(supertype(T))

Base.summary(io::IO, x::TensorNetwork) = print(io, "$(length(x))-tensors $(typeof(x))")
Base.show(io::IO, tn::TensorNetwork) =
    print(io, "$(typeof(tn))(#tensors=$(length(tn.tensors)), #inds=$(length(tn.indices)))")

"""
    copy(tn::TensorNetwork)

Return a shallow copy of the [`TensorNetwork`](@ref).
"""
Base.copy(tn::TensorNetwork{A}) where {A} = TensorNetwork{A}(copy(tn.tensors); deepcopy(tn.metadata)...)

"""
    ansatz(::TensorNetwork{Ansatz})
    ansatz(::Type{<:TensorNetwork{Ansatz}})

Return the `Ansatz` of a [`TensorNetwork`](@ref) type or object.
"""
ansatz(::Type{<:TensorNetwork{A}}) where {A} = A
ansatz(::TensorNetwork{A}) where {A} = A

"""
    tensors(tn::TensorNetwork)

Return a list of the `Tensor`s in the [`TensorNetwork`](@ref).
"""
tensors(tn::TensorNetwork) = tn.tensors
arrays(tn::TensorNetwork) = parent.(tensors(tn))

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
EinExprs.inds(tn::TensorNetwork; set::Symbol = :all, kwargs...) = inds(tn, set; kwargs...)
@valsplit 2 EinExprs.inds(tn::TensorNetwork, set::Symbol, args...) = throw(MethodError(inds, "set=$set not recognized"))
EinExprs.inds(tn::TensorNetwork, ::Val{:all}) = collect(keys(tn.indices))
EinExprs.inds(tn::TensorNetwork, ::Val{:open}) = map(first, Iterators.filter(==(1) ∘ length ∘ last, tn.indices))
EinExprs.inds(tn::TensorNetwork, ::Val{:inner}) = map(first, Iterators.filter(>=(2) ∘ length ∘ last, tn.indices))
EinExprs.inds(tn::TensorNetwork, ::Val{:hyper}) = map(first, Iterators.filter(>=(3) ∘ length ∘ last, tn.indices))

"""
    size(tn::TensorNetwork)
    size(tn::TensorNetwork, index)

Return a mapping from indices to their dimensionalities.

If `index` is set, return the dimensionality of `index`. This is equivalent to `size(tn)[index]`.
"""
Base.size(tn::TensorNetwork) = Dict(i => size(tn, i) for (i, x) in tn.indices)
Base.size(tn::TensorNetwork, i::Symbol) = size(tn.tensors[first(tn.indices[i])], i)

Base.eltype(tn::TensorNetwork) = promote_type(eltype.(tensors(tn))...)

Base.getindex(tn::TensorNetwork, key::Symbol) = tn.metadata[key]
Base.fieldnames(tn::T) where {T<:TensorNetwork} = fieldnames(T)
Base.propertynames(tn::TensorNetwork{A,N}) where {A,N} = tuple(fieldnames(tn)..., fieldnames(N)...)
Base.getproperty(tn::T, name::Symbol) where {T<:TensorNetwork} =
    if hasfield(T, name)
        getfield(tn, name)
    elseif hasfield(fieldtype(T, :metadata), name)
        getfield(getfield(tn, :metadata), name)
    else
        throw(KeyError(name))
    end

"""
    push!(tn::TensorNetwork, tensor::Tensor)

Add a new `tensor` to the Tensor Network.

See also: [`append!`](@ref), [`pop!`](@ref).
"""
function Base.push!(tn::TensorNetwork, tensor::Tensor)
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
    append!(tn::TensorNetwork, tensors::AbstractVecOrTuple{<:Tensor})
    append!(A::TensorNetwork, B::TensorNetwork)

Add a list of tensors to the first `TensorNetwork`.

See also: [`push!`](@ref)
"""
Base.append!(tn::TensorNetwork, t::AbstractVecOrTuple{<:Tensor}) = (foreach(Base.Fix1(push!, tn), t); tn)
function Base.append!(A::TensorNetwork, B::TensorNetwork)
    append!(A, tensors(B))
    # TODO define behaviour
    # merge!(A.metadata, B.metadata)
    return A
end

function Base.popat!(tn::TensorNetwork, i::Integer)
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
    pop!(tn::TensorNetwork, tensor::Tensor)
    pop!(tn::TensorNetwork, i::Union{Symbol,AbstractVecOrTuple{Symbol}})

Remove a tensor from the Tensor Network and returns it. If a `Tensor` is passed, then the first tensor satisfies _egality_ (i.e. `≡` or `===`) will be removed.
If a `Symbol` or a list of `Symbol`s is passed, then remove and return the tensors that contain all the indices.

See also: [`push!`](@ref), [`delete!`](@ref).
"""
function Base.pop!(tn::TensorNetwork, tensor::Tensor)
    i = findfirst(t -> t === tensor, tn.tensors)
    popat!(tn, i)
end

Base.pop!(tn::TensorNetwork, i::Symbol) = pop!(tn, (i,))

function Base.pop!(tn::TensorNetwork, i::AbstractVecOrTuple{Symbol})::Vector{Tensor}
    tensors = select(tn, i)
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

"""
    replace(tn::TensorNetwork, old => new...)

Return a copy of the [`TensorNetwork`](@ref) where `old` has been replaced by `new`.

See also: [`replace!`](@ref).
"""
Base.replace(tn::TensorNetwork, old_new::Pair...) = replace!(copy(tn), old_new...)

"""
    replace!(tn::TensorNetwork, old => new...)

Replace the element in `old` with the one in `new`. Depending on the types of `old` and `new`, the following behaviour is expected:

  - If `Symbol`s, it will correspond to a index renaming.
  - If `Tensor`s, first element that satisfies _egality_ (`≡` or `===`) will be replaced.

See also: [`replace`](@ref).
"""
function Base.replace!(tn::TensorNetwork, old_new::Pair...)
    for pair in old_new
        replace!(tn, pair)
    end
    return tn
end

function Base.replace!(tn::TensorNetwork, pair::Pair{<:Tensor,<:Tensor})
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

function Base.replace!(tn::TensorNetwork, old_new::Pair{Symbol,Symbol})
    old, new = old_new
    new ∈ inds(tn) && throw(ArgumentError("new symbol $new is already present"))

    push!(tn.indices, new => pop!(tn.indices, old))

    for i in tn.indices[new]
        tn.tensors[i] = replace(tn.tensors[i], old_new)
    end

    return tn
end

function Base.replace!(tn::TensorNetwork, old_new::Pair{<:Tensor,<:TensorNetwork})
    old, new = old_new
    issetequal(inds(new, set = :open), inds(old)) || throw(ArgumentError("indices must match"))

    # rename internal indices so there is no accidental hyperedge
    replace!(new, [index => Symbol(uuid4()) for index in filter(∈(inds(tn)), inds(new, set = :inner))]...)

    append!(tn, new)
    delete!(tn, old)

    return tn
end

"""
    select(tn::TensorNetwork, i)

Return tensors whose indices match with the list of indices `i`.
"""
select(tn::TensorNetwork, i::AbstractVecOrTuple{Symbol}) = filter(Base.Fix1(⊆, i) ∘ inds, tensors(tn))
select(tn::TensorNetwork, i::Symbol) = map(x -> tn.tensors[x], unique(tn.indices[i]))

"""
    in(tensor::Tensor, tn::TensorNetwork)

Return `true` if there is a `Tensor` in `tn` for which `==` evaluates to `true`.
This method is equivalent to `tensor ∈ tensors(tn)` code, but it's faster on large amount of tensors.
"""
Base.in(tensor::Tensor, tn::TensorNetwork) = in(tensor, select(tn, inds(tensor)))

"""
    slice!(tn::TensorNetwork, index::Symbol, i)

In-place projection of `index` on dimension `i`.

See also: [`selectdim`](@ref), [`view`](@ref).
"""
function slice!(tn::TensorNetwork, label::Symbol, i)
    for tensor in select(tn, label)
        pos = findfirst(t -> t === tensor, tn.tensors)
        tn.tensors[pos] = selectdim(tensor, label, i)
    end

    i isa Integer && delete!(tn.indices, label)

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
function Base.view(tn::TensorNetwork, slices::Pair{Symbol,<:Any}...)
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

    tensors = [Tensor(rand([size_dict[ind] for ind in input]...), tuple(input...)) for input in inputs]
    TensorNetwork(tensors)
end

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
    EinExpr(
        outputs,
        [EinExpr(inds(tensor), Dict(index => size(tensor, index) for index in inds(tensor))) for tensor in tensors(tn)],
    );
    kwargs...,
)

# TODO sequence of indices?
# TODO what if parallel neighbour indices?
"""
    contract!(tn::TensorNetwork, index)

In-place contraction of tensors connected to `index`.

See also: [`contract`](@ref).
"""
function contract!(tn::TensorNetwork, i)
    tensor = reduce(pop!(tn, i)) do acc, tensor
        contract(acc, tensor, dims = i)
    end

    push!(tn, tensor)
    return tn
end

"""
    contract(tn::TensorNetwork; kwargs...)

Contract a [`TensorNetwork`](@ref). The contraction order will be first computed by [`einexpr`](@ref).

The `kwargs` will be passed down to the [`einexpr`](@ref) function.

See also: [`einexpr`](@ref), [`contract!`](@ref).
"""
function contract(tn::TensorNetwork; path = einexpr(tn))
    # TODO does `first` work always?
    length(path.args) == 0 && return select(tn, inds(path)) |> first

    intermediates = map(subpath -> contract(tn; path = subpath), path.args)
    contract(intermediates...; dims = suminds(path))
end

contract!(t::Tensor, tn::TensorNetwork; kwargs...) = contract!(tn, t; kwargs...)
contract!(tn::TensorNetwork, t::Tensor; kwargs...) = (push!(tn, t); contract(tn; kwargs...))
contract(t::Tensor, tn::TensorNetwork; kwargs...) = contract(tn, t; kwargs...)
contract(tn::TensorNetwork, t::Tensor; kwargs...) = contract!(copy(tn), t; kwargs...)

struct TNSampler{A<:Ansatz,NT<:NamedTuple} <: Random.Sampler{TensorNetwork{A}}
    parameters::NT

    TNSampler{A}(; kwargs...) where {A} = new{A,typeof(values(kwargs))}(values(kwargs))
end

Base.getproperty(obj::TNSampler{A,<:NamedTuple{K}}, name::Symbol) where {A,K} =
    name ∈ K ? getfield(obj, :parameters)[name] : getfield(obj, name)
Base.get(obj::TNSampler, name, default) = get(getfield(obj, :parameters), name, default)

Base.eltype(::TNSampler{A}) where {A<:Ansatz} = TensorNetwork{A}

Base.rand(A::Type{<:Ansatz}; kwargs...) = rand(Random.default_rng(), A; kwargs...)
Base.rand(rng::AbstractRNG, ::Type{A}; kwargs...) where {A<:Ansatz} = rand(rng, TNSampler{A}(; kwargs...))

Base.convert(::Type{T}, tn::TensorNetwork{A}) where {T<:Ansatz,A<:T} =
    TensorNetwork{T}(tensors(tn); metadata(T)(tn.metadata)...)

Base.convert(::Type{T}, tn::TensorNetwork{A}; metadata...) where {A<:Ansatz,T<:A} = TensorNetwork{T}(tn; metadata...)

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
function checktopology end
function checkmeta end
function metadata end

"""
    TensorNetwork

Graph of interconnected tensors.

# Implementation details

`TensorNetwork` represents the **dual hypergraph** of the Tensor Network (i.e. vertices are indices and hyperedges are tensors).
"""
struct TensorNetwork{A<:Ansatz,M<:NamedTuple}
    indices::Dict{Symbol,Vector{Int}}
    tensors::Vector{Tensor}
    metadata::M

    function TensorNetwork{A}(tensors; metadata...) where {A}
        indices = reduce(enumerate(tensors); init = Dict{Symbol,Vector{Int}}([])) do dict, (i, tensor)
            mergewith(vcat, dict, Dict([index => [i] for index in labels(tensor)]))
        end

        M = merge(Tenet.metadata.(superansatzes(A))...)
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
    checktopology(tn) || throw(ErrorException("\"$A\" topology not preserved"))

    for T in superansatzes(A)
        checkmeta(T, tn) || throw(ErrorException("\"$T\" metadata is not valid"))
    end
end

checktopology(::TensorNetwork{<:Ansatz}) = true
checkmeta(::Type{<:Ansatz}, ::TensorNetwork) = true
checkmeta(tn::TensorNetwork{T}) where {T<:Ansatz} = all(A -> checkmeta(A, tn), superansatzes(T))

metadata(::Type{<:Ansatz}) = NamedTuple{(),Tuple{}}

Base.summary(io::IO, x::TensorNetwork) = print(io, "$(length(x))-tensors $(typeof(x))")
Base.show(io::IO, tn::TensorNetwork) = print(io, "$(typeof(tn))(#tensors=$(length(tn)), #labels=$(length(tn.indices)))")
Base.length(x::TensorNetwork) = length(tensors(x))

Base.copy(tn::TensorNetwork{A}) where {A} = TensorNetwork{A}(copy(tn.tensors); deepcopy(tn.metadata)...)

ansatz(::Type{<:TensorNetwork{A}}) where {A} = A
ansatz(::TensorNetwork{A}) where {A} = A

tensors(tn::TensorNetwork) = tn.tensors
arrays(tn::TensorNetwork) = parent.(tensors(tn))

labels(tn::TensorNetwork; set::Symbol = :all, kwargs...) = labels(tn, set; kwargs...)
@valsplit 2 labels(tn::TensorNetwork, set::Symbol, args...) = throw(MethodError(labels, "set=$set not recognized"))
labels(tn::TensorNetwork, ::Val{:all}) = collect(keys(tn.indices))
labels(tn::TensorNetwork, ::Val{:open}) = map(first, Iterators.filter(==(1) ∘ length ∘ last, tn.indices))
labels(tn::TensorNetwork, ::Val{:inner}) = map(first, Iterators.filter(>=(2) ∘ length ∘ last, tn.indices))
labels(tn::TensorNetwork, ::Val{:hyper}) = map(first, Iterators.filter(>=(3) ∘ length ∘ last, tn.indices))

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

function Base.push!(tn::TensorNetwork, tensor::Tensor)
    for i in Iterators.filter(i -> size(tn, i) != size(tensor, i), labels(tensor) ∩ labels(tn))
        throw(DimensionMismatch("size(tensor,$i)=$(size(tensor,i)) but should be equal to size(tn,$i)=$(size(tn,i))"))
    end

    push!(tn.tensors, tensor)

    for i in labels(tensor)
        push!(get!(tn.indices, i, Int[]), length(tn.tensors))
    end

    return tn
end

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
    for index in unique(labels(tensor))
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

Base.delete!(tn::TensorNetwork, x) = (_ = pop!(tn, x); tn)

Base.replace(tn::TensorNetwork, old_new::Pair...) = replace!(copy(tn), old_new...)

function Base.replace!(tn::TensorNetwork, pair::Pair{<:Tensor,<:Tensor})
    old_tensor, new_tensor = pair

    # check if old and new tensors are compatible
    if !issetequal(labels(new_tensor), labels(old_tensor))
        throw(ArgumentError("New tensor labels do not match the existing tensor labels"))
    end

    # replace existing `Tensor` with new `Tensor`
    i = findfirst(t -> t === old_tensor, tn.tensors)
    splice!(tn.tensors, i, [new_tensor])

    return tn
end

function Base.replace!(tn::TensorNetwork, old_new::Pair{Symbol,Symbol})
    old, new = old_new
    new ∈ labels(tn) && throw(ArgumentError("new symbol $new is already present"))

    push!(tn.indices, new => pop!(tn.indices, old))

    for i in tn.indices[new]
        tn.tensors[i] = replace(tn.tensors[i], old_new)
    end

    return tn
end

function Base.replace!(tn::TensorNetwork, old_new::Pair...)
    for pair in old_new
        replace!(tn, pair)
    end
    return tn
end

function Base.replace!(tn::TensorNetwork, old_new::Pair{<:Tensor,<:TensorNetwork})
    old, new = old_new
    issetequal(labels(new, set = :open), labels(old)) || throw(ArgumentError("indices must match"))

    # rename internal indices so there is no accidental hyperedge
    replace!(new, [index => Symbol(uuid4()) for index in filter(∈(labels(tn)), labels(new, set = :inner))]...)

    append!(tn, new)
    delete!(tn, old)

    return tn
end

"""
    select(tn, i)

Return tensors whose labels match with the list of indices `i`.
"""
select(tn::TensorNetwork, i::AbstractVecOrTuple{Symbol}) = mapreduce(Base.Fix1(select, tn), ∩, i)
select(tn::TensorNetwork, i::Symbol) = map(x -> tn.tensors[x], unique(tn.indices[i]))

"""
    slice!(tn, index, i)

In-place slice `index` of the Tensor Network `tn` on dimension `i`.
"""
function slice!(tn::TensorNetwork, label::Symbol, i)
    for tensor in select(tn, label)
        pos = findfirst(t -> t === tensor, tn.tensors)
        tn.tensors[pos] = selectdim(tensor, label, i)
    end

    i isa Integer && delete!(tn.indices, label)

    return tn
end

Base.selectdim(tn::TensorNetwork, label::Symbol, i) = @view tn[label=>i]
function Base.view(tn::TensorNetwork, slices::Pair{Symbol,<:Any}...)
    tn = copy(tn)

    for (label, i) in slices
        slice!(tn, label, i)
    end

    return tn
end

"""
    rand(TensorNetwork, n, regularity[, out = 0, dim = 2:9, seed = nothing, globalind = false])

Generate a random contraction and shapes.

# Arguments

  - `n`: Number of array arguments.
  - `regularity`: 'Regularity' of the contraction graph. This essentially determines how many indices each tensor shares with others on average.
  - `out=0`: Number of output indices (i.e. the number of non-contracted indices).
  - `dim=2:9`: Range of dimension sizes.
  - `seed=nothing`: If not `nothing`, seed random generator with this value.
  - `globalind=false`: Add a global, 'broadcast', dimension to every tensor.
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

EinExprs.einexpr(tn::TensorNetwork; optimizer = Greedy, outputs = labels(tn, :open), kwargs...) =
    einexpr(optimizer, EinExpr(tensors(tn), outputs); kwargs...)

# TODO sequence of indices?
# TODO what if parallel neighbour indices?
function contract!(tn::TensorNetwork, i::Symbol)
    tensor = reduce(pop!(tn, i)) do acc, tensor
        contract(acc, tensor, i)
    end

    push!(tn, tensor)
    return tn
end

contract(tn::TensorNetwork; outputs = labels(tn, :open), kwargs...) =
    contract(einexpr(tn; outputs = outputs, kwargs...))

contract(t::Tensor, tn::TensorNetwork; kwargs...) = contract(tn, t; kwargs...)
contract(tn::TensorNetwork, t::Tensor; kwargs...) = (tn = copy(tn); append!(tn, t); contract(tn; kwargs...))

struct TNSampler{A<:Ansatz,NT<:NamedTuple} <: Random.Sampler{TensorNetwork{A}}
    parameters::NT

    TNSampler{A}(; kwargs...) where {A} = new{A,typeof(values(kwargs))}(values(kwargs))
end

Base.getproperty(obj::TNSampler{A,<:NamedTuple{K}}, name::Symbol) where {A,K} =
    name ∈ K ? getfield(obj, :parameters)[name] : getfield(obj, name)
Base.get(obj::TNSampler, name, default) = get(getfield(obj, :parameters), name, default)

Base.eltype(::TNSampler{A}) where {A<:Ansatz} = TensorNetwork{A}

Base.rand(::Type{A}; kwargs...) where {A<:Ansatz} = rand(TNSampler{A}(; kwargs...))

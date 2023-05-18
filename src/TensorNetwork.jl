using Random
using EinExprs
using OMEinsum

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

"""
    TensorNetwork

Graph of interconnected tensors.
"""
struct TensorNetwork{A}
    tensors::Vector{Tensor}
    inds::Dict{Symbol,Index}
    meta::Dict{Symbol,Any}

    function TensorNetwork{A}(; meta...) where {A}
        meta = Dict{Symbol,Any}(meta)

        new{A}(Tensor[], Dict{Symbol,Index}(), meta)
    end
end

TensorNetwork(args...; kwargs...) = TensorNetwork{Arbitrary}(args...; kwargs...)
function TensorNetwork{A}(tensors; meta...) where {A}
    tn = TensorNetwork{A}(; meta...)

    for tensor in tensors
        push!(tn, tensor)
    end

    return tn
end

# TODO checks? index metadata?
TensorNetwork{A}(tn::TensorNetwork{B}) where {A,B} = TensorNetwork{B}(tensors(tn); tn.meta...)

Base.summary(io::IO, x::TensorNetwork) = print(io, "$(length(x))-tensors $(typeof(x))")
Base.show(io::IO, tn::TensorNetwork) = print(io, "$(typeof(tn))(#tensors=$(length(tn)), #inds=$(length(inds(tn))))")
Base.length(x::TensorNetwork) = length(tensors(x))

function Base.copy(tn::TensorNetwork{A}) where {A}
    newtn = TensorNetwork{A}(; copy(tn.meta)...)
    append!(newtn, copy(tn.tensors))
    for i in labels(newtn)
        merge!(newtn.inds[i].meta, copy(tn.inds[i].meta))
    end
    return newtn
end

ansatz(::Type{TensorNetwork{A}}) where {A} = A
ansatz(::TensorNetwork{A}) where {A} = A

function tensors end
tensors(tn::TensorNetwork) = tn.tensors
tensors(tn::TensorNetwork, i::Integer) = tn.tensors[i]
tensors(tn::TensorNetwork, i::Symbol)::Vector{Tensor} = links(tn.inds[i])
tensors(tn::TensorNetwork, i::Index)::Vector{Tensor} = tensors(tn, nameof(i))
tensors(tn::TensorNetwork, inds::Base.AbstractVecOrTuple{Union{Symbol,Index}}) = ∩([tensors(tn, i) for i in inds]...)
arrays(tn::TensorNetwork) = parent.(tensors(tn))

function inds end
inds(tn::TensorNetwork) = values(tn.inds)
inds(tn::TensorNetwork, label) = tn.inds[label]
labels(tn::TensorNetwork) = nameof.(inds(tn))

openinds(tn::TensorNetwork) = Iterators.filter(isopenind, inds(tn)) |> collect
innerinds(tn::TensorNetwork) = Iterators.filter(!isopenind, inds(tn)) |> collect
hyperinds(tn::TensorNetwork) = Iterators.filter(ishyperind, inds(tn)) |> collect

Base.size(tn::TensorNetwork) = Dict(nameof(i) => size(i) for i in inds(tn))
Base.size(tn::TensorNetwork, i::Symbol) = size(tn.inds[i])

Base.eltype(tn::TensorNetwork) = promote_type(eltype.(tensors(tn))...)

"""
    select(tn, i)

Return tensors whose labels match with the list of indices `i`.
"""
select(tn::TensorNetwork, i::Base.AbstractVecOrTuple{Symbol}) = ∩(map(Base.Fix1(select, tn), i)...)
select(tn::TensorNetwork, i::Symbol) = links(tn.inds[i])

"""
    selectdim(tn, index, i)

Returns a view of the Tensor Network `tn` of slice `i` on `index`.
"""
Base.selectdim(tn::TensorNetwork, label::Symbol, i) = selectdim!(copy(tn), label, i)

function selectdim!(tn::TensorNetwork, label::Symbol, i)
    # UNSAFE REGION BEGIN
    if !isa(i, Integer)
        index = tn.inds[label]
        tn.inds[label] = Index(nameof(index), length(i); index.meta...)
    end
    # UNSAFE REGION END

    for tensor in links(tn.inds[label])
        push!(tn, selectdim(tensor, label, i))
        delete!(tn, tensor)
    end

    return tn
end

function Base.view(tn::TensorNetwork, inds::Pair{Symbol,<:Any}...)
    tn = copy(tn)

    for (label, i) in inds
        selectdim!(tn, label, i)
    end

    return tn
end

function Base.intersect(a::TensorNetwork, b::TensorNetwork{A}) where {A}
    # TODO add index metadata?
    # TODO :plug?
    c = TensorNetwork{A}(filter(tensors(a)) do tensor
        !isdisjoint(labels(tensor), labels(b))
    end; a.meta...)

    return c
end

function Base.push!(tn::TensorNetwork, tensor::Tensor)
    push!(tensors(tn), tensor)

    # TODO merge metadata?
    for (i, s) in zip(labels(tensor), size(tensor))
        if i ∉ keys(tn.inds)
            tn.inds[i] = Index(i, s)
        end

        if s != size(tn.inds[i])
            throw(DimensionMismatch("size($i)=$s but should be $(size(tn.inds[i]))"))
        end

        link!(tn.inds[i], tensor)
    end

    return tn
end

Base.append!(tn::TensorNetwork, t::Base.AbstractVecOrTuple{<:Tensor}) = (foreach(Base.Fix1(push!, tn), t); tn)
function Base.append!(A::TensorNetwork, B::TensorNetwork)
    append!(A, tensors(B))

    # merge index metadata
    for i in labels(B)
        merge!(A.inds[i].meta, copy(B.inds[i].meta))
    end

    return A
end

function Base.popat!(tn::TensorNetwork, i::Integer)
    tensor = popat!(tensors(tn), i)

    # unlink indices
    for i in unique(labels(tensor))
        index = tn.inds[i]
        unlink!(index, tensor)

        # remove index when no tensors plugged in
        length(links(index)) == 0 && delete!(tn.inds, i)
    end

    return tensor
end

function Base.pop!(tn::TensorNetwork, tensor::Tensor)
    i = findfirst(t -> t === tensor, tn.tensors)
    popat!(tn, i)
end

Base.pop!(tn::TensorNetwork, i::Symbol) = pop!(tn, (i,))

function Base.pop!(tn::TensorNetwork, i::Base.AbstractVecOrTuple{Symbol})::Vector{Tensor}
    tensors = select(tn, i)
    for tensor in tensors
        _ = pop!(tn, tensor)
    end

    return tensors
end

Base.delete!(tn::TensorNetwork, x) = (_ = pop!(tn, x); tn)

Base.replace(tn::TensorNetwork, old_new::Pair...) = replace!(copy(tn), old_new...)

function Base.replace!(tn::TensorNetwork, old_new::Pair{<:Tensor,<:Tensor}...)
    # check if new tensors are already present in the network
    new_tensors = last.(old_new)
    isdisjoint(new_tensors, tn.tensors) ||
        throw(ArgumentError("New tensors must not be already present in the network"))

    # check if old and new tensors are compatible
    all(pair -> issetequal(map(labels, pair)...), old_new) ||
        throw(ArgumentError("New tensor labels do not match the existing tensor labels"))

    for (old_tensor, new_tensor) in old_new
        # update index links
        # TODO remove this part when `Index` is removed
        for old_label in labels(old_tensor)
            index_obj = tn.inds[old_label]
            link_index = findfirst(x -> x === old_tensor, index_obj.links)
            index_obj.links[link_index] = new_tensor
        end

        # replace existing `Tensor` with new `Tensor`
        index = findfirst(x -> x === old_tensor, tn.tensors)
        tn.tensors[index] = new_tensor
    end

    return tn
end

function Base.replace!(tn::TensorNetwork, old_new::Pair{Symbol,Symbol}...)
    isdisjoint(last.(old_new), labels(tn)) || throw(ArgumentError("target symbols must not be already present"))

    tensors = unique(Iterators.flatten([select(tn, i) for i in first.(old_new)]) |> collect)

    # reindex indices
    # NOTE what if :a => :d, :d => ...? temporal location as a fix
    tmp = Dict(name => copy(i) for (name, i) in tn.inds if name ∈ first.(old_new))
    for (old, new) in old_new
        push!(tn.inds, new => replace(tmp[old], new))
    end

    # reindex tensors
    for tensor in tensors
        push!(tn, replace(tensor, old_new...))
        delete!(tn, tensor)
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
function Random.rand(
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

EinExprs.einexpr(tn::TensorNetwork; optimizer = Greedy, outputs = openinds(tn), kwargs...) =
    einexpr(optimizer, EinExpr(tensors(tn), outputs); kwargs...)

# TODO sequence of indices?
# TODO what if parallel neighbour indices?
function contract!(tn::TensorNetwork, i::Symbol)
    ts = pop!(tn, i) # map(Base.Fix1(pop!, tn), links(index))

    tensor = reduce((acc, t) -> contract(acc, t, i), ts)

    # NOTE index is automatically cleaned
    push!(tn, tensor)
end

contract(tn::TensorNetwork; outputs = openinds(tn), kwargs...) = contract(einexpr(tn; outputs = outputs, kwargs...))

contract(t::Tensor, tn::TensorNetwork; kwargs...) = contract(tn, t; kwargs...)
contract(tn::TensorNetwork, t::Tensor; kwargs...) = (tn = copy(tn); append!(tn, t); contract(tn; kwargs...))

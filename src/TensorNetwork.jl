import Random: rand
import OptimizedEinsum
import OptimizedEinsum: contractpath, Solver, Greedy, ContractionPath
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
struct TensorNetwork{A<:Ansatz}
    tensors::Vector{Tensor}
    inds::Dict{Symbol,Index}
    meta::Dict{Symbol,Any}

    function TensorNetwork{A}(; meta...) where {A<:Ansatz}
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

Base.summary(io::IO, x::TensorNetwork) = print(io, "$(length(x))-tensors $(typeof(x))")
Base.show(io::IO, tn::TensorNetwork) = print(io, "$(typeof(tn))(#tensors=$(length(tn)), #inds=$(length(inds(tn))))")
Base.length(x::TensorNetwork) = length(tensors(x))

function Base.copy(tn::TensorNetwork{A}) where {A<:Ansatz}
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
tensors(tn::TensorNetwork, inds::Sequence{Union{Symbol,Index}}) = ∩([tensors(tn, i) for i in inds]...)
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
select(tn::TensorNetwork, i::Sequence{Symbol}) = ∩(map(Base.Fix1(select, tn), i)...)
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

Base.append!(tn::TensorNetwork, t::Sequence{<:Tensor}) = (foreach(Base.Fix1(push!, tn), t); tn)
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

function Base.pop!(tn::TensorNetwork, i::Sequence{Symbol})::Vector{Tensor}
    tensors = select(tn, i)
    for tensor in tensors
        _ = pop!(tn, tensor)
    end

    return tensors
end

Base.delete!(tn::TensorNetwork, x) = (_ = pop!(tn, x); tn)

Base.replace(tn::TensorNetwork, old_new::Pair{Symbol,Symbol}...) = replace!(copy(tn), old_new...)

function Base.replace!(tn::TensorNetwork, old_new::Pair{Symbol,Symbol}...)
    !isdisjoint(values(old_new), labels(tn)) && throw(ArgumentError("target symbols must not be already present"))

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

function rand(::Type{TensorNetwork}, n::Integer, reg::Integer; kwargs...)
    # TODO `output` is not used
    output, inputs, size_dict = OptimizedEinsum.rand_equation(n, reg, kwargs...)
    tensors = [Tensor(rand([size_dict[ind] for ind in input]...), tuple(input...)) for input in inputs]
    TensorNetwork(tensors)
end

function contractpath(tn::TensorNetwork; solver = Greedy, output = openinds(tn), kwargs...)
    inputs = collect.(labels.(tensors(tn)))
    output = collect(nameof.(output))
    size_dict = size(tn)

    contractpath(solver, inputs, output, size_dict)
end

# TODO sequence of indices?
# TODO what if parallel neighbour indices?
function contract!(tn::TensorNetwork, i::Symbol)
    ts = pop!(tn, i) # map(Base.Fix1(pop!, tn), links(index))

    tensor = reduce((acc, t) -> contract(acc, t, i), ts)

    # NOTE index is automatically cleaned
    push!(tn, tensor)
end

function contract(tn::TensorNetwork; output = openinds(tn), kwargs...)
    path = contractpath(tn; output = output, kwargs...)

    # SSA-to-tensor mapping
    mapping = Dict{Int,Tensor}(i => t for (i, t) in enumerate(tensors(tn)))

    for (c, (a, b)) in zip(Iterators.countfrom(length(path.inputs) + 1), path)
        A = pop!(mapping, a)
        B = pop!(mapping, b)

        indsA = labels(A)
        indsB = labels(B)
        indsC = symdiff(indsA, indsB) ∪ ∩(output, indsA, indsB)

        C = EinCode((map(String, indsA), map(String, indsB)), tuple(map(String, indsC)...))(A, B)

        mapping[c] = Tensor(C, tuple(indsC...))
    end

    only(values(mapping))
end

contract(t::Tensor, tn::TensorNetwork; kwargs...) = contract(tn, t; kwargs...)
contract(tn::TensorNetwork, t::Tensor; kwargs...) = (tn = copy(tn); append!(tn, t); contract(tn; kwargs...))

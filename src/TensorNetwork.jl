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
    Quantum

Tensor Networks that have a notion of site and direction (input/output).
"""
abstract type Quantum <: Ansatz end

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
    # NOTE calling `copy` on each tensor, so tensors are unlinked
    tensors = copy.(tensors)
    tn = TensorNetwork{A}(; meta...)

    for tensor in tensors
        push!(tn, tensor)
    end

    return tn
end

Base.summary(io::IO, x::TensorNetwork) = print(io, "$(length(x))-tensors $(typeof(x))")
Base.show(io::IO, tn::TensorNetwork) = print(io, "$(typeof(tn))(#tensors=$(length(tn)), #inds=$(length(inds(tn))))")
Base.length(x::TensorNetwork) = length(tensors(x))

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
labels(tn::TensorNetwork) = nameof.(inds(tn))

openinds(tn::TensorNetwork) = Iterators.filter(isopenind, inds(tn)) |> collect
hyperinds(tn::TensorNetwork) = Iterators.filter(ishyperind, inds(tn)) |> collect
physicalinds(tn::TensorNetwork) = Iterators.filter(isphysical, inds(tn)) |> collect
virtualinds(tn::TensorNetwork) = Iterators.filter(isvirtual, inds(tn)) |> collect

Base.size(tn::TensorNetwork) = Dict(nameof(i) => size(i) for i in inds(tn))
Base.size(tn::TensorNetwork, i::Symbol) = size(tn.inds[i])

"""
    select(tn, i)

Return tensors whose labels match with the list of indices `i`.
"""
select(tn::TensorNetwork, i::Sequence{Symbol}) = ∩(map(Base.Fix1(select, tn), labels)...)
select(tn::TensorNetwork, i::Symbol) = links(tn.inds[i])

function Base.push!(tn::TensorNetwork, tensor::Tensor)
    push!(tensors(tn), tensor)

    # TODO merge metadata?
    for i in labels(tensor)
        if i ∉ keys(tn.inds)
            tn.inds[i] = Index(i, size(tensor, i))
        end

        link!(tn.inds[i], tensor)
    end

    return tn
end

Base.append!(tn::TensorNetwork, t::Sequence{Tensor}) = (foreach(Base.Fix1(push!, A), t); tn)
Base.append!(A::TensorNetwork, B::TensorNetwork) = append!(A, tensors(B))

function Base.popat!(tn::TensorNetwork, i::Integer)
    tensor = popat!(tensors(tn), i)

    # unlink indices
    for i in inds(tensor)
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
    foreach(Base.Fix1(pop!, tn), tensors)

    return tensors
end

Base.delete!(tn::TensorNetwork, x) = (_ = pop!(tn, x); tn)

function rand(::Type{TensorNetwork}, n::Integer, reg::Integer; kwargs...)
    output, inputs, size_dict = OptimizedEinsum.rand_equation(n, reg, kwargs...)
    tensors = [Tensor(rand([size_dict[ind] for ind in input]...), tuple(input...)) for input in inputs]
    TensorNetwork(tensors)
end

function contractpath(tn::TensorNetwork; solver = Greedy, output = openinds(tn), kwargs...)
    inputs = collect.(labels.(tensors(tn)))
    output = collect(output)
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
    path = OptimizedEinsum.contractpath(tn; output = output, kwargs...)

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

using Quac: Circuit, lanes, arraytype, Swap
using OptimizedEinsum: get_symbol
using LinearAlgebra
using UUIDs: uuid4

"""
    Quantum

Tensor Network [`Ansatz`](@ref) that has a notion of sites and directionality (input/output).
"""
abstract type Quantum <: Ansatz end

abstract type Bounds end
abstract type Closed <: Bounds end
abstract type Open <: Bounds end

"""
    State{Bounds}

[`Quantum`](@ref) Tensor Network that only has outputs. Usually, it reflects the _state_ of a physical system.

Its adjoint only has inputs.
"""
abstract type State{B} <: Quantum where {B<:Bounds} end
bounds(::T) where {T<:State} = bounds(T)
bounds(::Type{<:State{B}}) where {B} = B

"""
    Operator{Bounds}

[`Quantum`](@ref) Tensor Network that has both inputs and outputs. Generally, it represents evolutionary processes of physical systems.
"""
abstract type Operator{B} <: Quantum where {B<:Bounds} end
bounds(::T) where {T<:Operator} = bounds(T)
bounds(::Type{<:Operator{B}}) where {B} = B

sites(tn::TensorNetwork) = insites(tn) ∪ outsites(tn)
# sites(tn::TensorNetwork{<:State}) = outsites(tn)
# sites(tn::Adjoint{TensorNetwork{<:State}}) = insites(tn)
siteinds(tn::TensorNetwork) = insiteinds(tn) ∪ outsiteinds(tn)

# TODO maybe don't filter by openinds?
insites(tn::TensorNetwork) = site.(insiteinds(tn))
# insites(::TensorNetwork{<:State}) = throw(MethodError(insites, TensorNetwork{<:State}))
# insites(tn::Adjoint{TensorNetwork}) = outsites(parent(tn))
insiteinds(tn) = sort!(filter(i -> get(i.meta, :plug, nothing) == :input, openinds(tn)), by = site)
insiteind(tn, s) = only(filter(i -> site(i) == s, insiteinds(tn)))

# TODO maybe don't filter by openinds?
outsites(tn::TensorNetwork) = site.(outsiteinds(tn))
# outsites(tn::Adjoint{TensorNetwork}) = insites(parent(tn))
outsiteinds(tn) = sort!(filter(i -> get(i.meta, :plug, nothing) == :output, openinds(tn)), by = site)
outsiteind(tn, s) = only(filter(i -> site(i) == s, outsiteinds(tn)))

physicalinds(tn::TensorNetwork) = Iterators.filter(isphysical, inds(tn)) |> collect
virtualinds(tn::TensorNetwork) = Iterators.filter(isvirtual, inds(tn)) |> collect

function TensorNetwork(circuit::Circuit)
    tn = TensorNetwork{Quantum}()
    n = lanes(circuit)

    wire = [[get_symbol(i)] for i in 1:n]
    i = n + 1

    for gate in circuit
        G = arraytype(gate)
        array = G(gate)

        if gate isa Swap
            (a, b) = lanes(gate)
            wire[a], wire[b] = wire[b], wire[a]
            continue
        end

        inds = map(lanes(gate)) do l
            from, to = last(wire[l]), get_symbol(i)
            i += 1
            push!(wire[l], to)
            (from, to)
        end |> x -> zip(x...) |> Iterators.flatten |> collect

        tensor = Tensor(array, tuple(inds...); gate = gate)
        push!(tn, tensor)
    end

    for (lane, wireindices) in enumerate(wire)
        for index in wireindices
            tn.inds[index].meta[:site] = lane
        end
    end

    for input in first.(wire)
        tn.inds[input].meta[:plug] = :input
    end

    for output in last.(wire)
        tn.inds[output].meta[:plug] = :output
    end

    return tn
end

function Base.hcat(A::TensorNetwork{QA}, B::TensorNetwork{QB}) where {QA<:Quantum,QB<:Quantum}
    outsites(A) != insites(B) && throw(DimensionMismatch("insites(B) must be equal to outsites(A) to connect them"))

    # rename connector indices
    newinds = Dict([s => Symbol(uuid4()) for s in outsites(A)])

    A = replace(A, [nameof(i) => newinds[site(i)] for i in outsiteinds(A)]...)
    B = replace(B, [nameof(i) => newinds[site(i)] for i in insiteinds(B)]...)

    # remove plug metadata on connector indices
    for i in values(newinds)
        delete!(A.inds[i].meta, :plug)
        delete!(B.inds[i].meta, :plug)
    end

    # rename inner indices of B to avoid hyperindices
    replace!(B, [nameof(i) => Symbol(uuid4()) for i in innerinds(B)]...)

    # merge tensors and indices
    tn = TensorNetwork{Tuple{QA,QB}}(; merge(A.meta, B.meta)...)
    append!(tn, A)
    append!(tn, B)

    return tn
end

Base.hcat(tns::TensorNetwork...) = reduce(hcat, tns)

function Base.adjoint(tn::TensorNetwork{A}) where {A<:Quantum}
    tn = deepcopy(tn)

    # TODO refactor internals
    for i in siteinds(tn)
        plug = i.meta[:plug]
        i.meta[:plug] = if plug == :input
            :output
        elseif plug == :output
            :input
        else
            # TODO throw error?
        end
    end

    for tensor in tensors(tn)
        tensor .= conj(tensor)
    end

    return tn
end

function contract(a::TensorNetwork{<:Quantum}, b::TensorNetwork{<:Quantum}; kwargs...)
    contract(hcat(a, b); kwargs...)
end

# TODO look for more stable ways
function LinearAlgebra.norm(ψ::TensorNetwork{<:State}, p::Real = 2; kwargs...)
    p != 2 && throw(ArgumentError("p=$p is not implemented yet"))

    return contract(hcat(ψ, ψ'); kwargs...) |> only |> sqrt |> abs
end

function LinearAlgebra.normalize!(
    ψ::TensorNetwork{<:State},
    p::Real = 2;
    insert::Union{Nothing,Int} = nothing,
    kwargs...,
)
    norm = LinearAlgebra.norm(ψ; kwargs...)

    if isnothing(insert)
        # method 1: divide all tensors by (√v)^(1/n)
        n = length(ψ)
        norm ^= 1 / n
        for tensor in tensors(ψ)
            tensor ./= norm
        end
    else
        # method 2: divide only one tensor
        tensor = tensors(ψ, insert)
        tensor ./= norm
    end
end

fidelity(a, b; kwargs...) = abs(only(contract(a, b'; kwargs...)))^2

tensors(tn::TensorNetwork{<:Quantum}, i::Integer) =
    only(tensors(tn, first(Iterators.filter(p -> site(p[2]) == i, tn.inds))[2]))

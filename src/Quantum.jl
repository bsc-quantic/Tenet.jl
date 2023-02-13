using Quac: Circuit, lanes, arraytype, Swap
using OptimizedEinsum: get_symbol
using LinearAlgebra: Adjoint
using UUIDs: uuid4

"""
    Quantum

Tensor Networks that have a notion of site and direction (input/output).
"""
abstract type Quantum <: Ansatz end

abstract type Bounds end
abstract type Closed <: Bounds end
abstract type Open <: Bounds end

"""
    State

Tensor Networks that only have outputs. Usually, they reflect the _state_ of physical systems.

Its adjoints only have inputs.
"""
abstract type State{B} <: Quantum where {B<:Bounds} end
bounds(::T) where {T<:State} = bounds(T)
bounds(::Type{<:State{B}}) where {B} = B

"""
    Operator

Tensor Networks that have both inputs and outputs. Generally, they represent evolutionary processes of physical systems.
"""
abstract type Operator{B} <: Quantum where {B<:Bounds} end
bounds(::T) where {T<:Operator} = bounds(T)
bounds(::Type{<:Operator{B}}) where {B} = B

sites(tn::TensorNetwork) = insites(tn) ∪ outsites(tn)
sites(tn::TensorNetwork{<:State}) = outsites(tn)
sites(tn::Adjoint{TensorNetwork{<:State}}) = insites(tn)

# TODO maybe don't filter by openinds?
insites(tn::TensorNetwork) = Set(site.(insiteinds(tn)))
insites(::TensorNetwork{<:State}) = throw(MethodError(insites, TensorNetwork{<:State}))
insites(tn::Adjoint{TensorNetwork}) = outsites(parent(tn))
insiteinds(tn) = filter(i -> i.meta[:plug] == :input, openinds(tn))

# TODO maybe don't filter by openinds?
outsites(tn::TensorNetwork) = Set(site.(outsiteinds(tn)))
outsites(tn::Adjoint{TensorNetwork}) = insites(parent(tn))
outsiteinds(tn) = filter(i -> i.meta[:plug] == :output, openinds(tn))

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

function Base.hcat(A::TensorNetwork{<:Quantum}, B::TensorNetwork{<:Quantum})
    outsites(A) != insites(B) && throw(DimensionMismatch("insites(B) must be equal to outsites(A) to connect them"))

    A = copy(A)
    B = copy(B)

    # rename connector indices
    newinds = Dict([s => Symbol(uuid4()) for s in outsites(A)])

    replace!(A, [nameof(i) => newinds[site(i)] for i in outsiteinds(A)]...)
    replace!(B, [nameof(i) => newinds[site(i)] for i in insiteinds(B)]...)

    # remove plug metadata on connector indices
    for i in values(newinds)
        delete!(A.inds[i].meta, :plug)
        delete!(B.inds[i].meta, :plug)
    end

    # rename inner indices of B to avoid hyperindices
    replace!(B, [nameof(i) => Symbol(uuid4()) for i in innerinds(B)]...)

    # merge tensors and indices
    append!(A, B)

    return A
end

Base.hcat(tns::TensorNetwork...) = reduce(hcat, tns)
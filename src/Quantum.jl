using Quac: Circuit, lanes, arraytype, Swap
using OptimizedEinsum: get_symbol
using LinearAlgebra: Adjoint

"""
    Quantum

Tensor Networks that have a notion of site and direction (input/output).
"""
abstract type Quantum <: Ansatz end

"""
    State

Tensor Networks that only have outputs. Usually, they reflect the _state_ of physical systems.

Its adjoints only have inputs.
"""
abstract type State <: Quantum end

"""
    Operator

Tensor Networks that have both inputs and outputs. Generally, they represent evolutionary processes of physical systems.
"""
abstract type Operator <: Quantum end

sites(tn::TensorNetwork) = insites(tn) ∪ outsites(tn)

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

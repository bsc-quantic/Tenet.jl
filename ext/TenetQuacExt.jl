module TenetQuacExt

using Tenet
using Quac: Circuit, lanes, arraytype, Swap

function Tenet.QuantumTensorNetwork(circuit::Circuit)
    n = lanes(circuit)
    wire = [[Tenet.letter(i)] for i in 1:n]
    tn = TensorNetwork()

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
            from, to = last(wire[l]), Tenet.letter(i)
            i += 1
            push!(wire[l], to)
            (from, to)
        end |> x -> zip(x...) |> Iterators.flatten |> collect

        tensor = Tensor(array, inds)
        push!(tn, tensor)
    end

    input = first.(wire)
    output = last.(wire)

    return QuantumTensorNetwork(tn, input, output)
end

end

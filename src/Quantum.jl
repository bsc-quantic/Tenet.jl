using Quac
using OptimizedEinsum: get_symbol

function TensorNetwork(circuit::Circuit)
    tn = TensorNetwork()
    n = lanes(circuit)

    wire = [get_symbol(i) for i in 1:n]
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
            from, to = wire[l], get_symbol(i)
            i += 1
            wire[l] = to
            (from, to)
        end |> x -> zip(x...) |> Iterators.flatten |> collect

        tensor = Tensor(array, tuple(inds...))
        push!(tn, tensor)
    end

    return tn
end
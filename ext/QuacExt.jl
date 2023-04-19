module QuacExt

if isdefined(Base, :get_extension)
    using Tenet
else
    using ..Tenet
end

using Quac: Circuit, lanes, arraytype, Swap
using OptimizedEinsum: get_symbol

function Tenet.TensorNetwork(circuit::Circuit)
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

end
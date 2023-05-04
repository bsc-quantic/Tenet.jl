module QuacExt

if isdefined(Base, :get_extension)
    using Tenet
else
    using ..Tenet
end

using Quac: Circuit, lanes, arraytype, Swap

function Tenet.TensorNetwork(circuit::Circuit)
    tn = TensorNetwork{Quantum}(; plug = Dict{Tuple{Int,Symbol},Symbol}())
    n = lanes(circuit)

    wire = [[Tenet.letter(i)] for i in 1:n]
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

        tensor = Tensor(array, tuple(inds...); gate = gate)
        push!(tn, tensor)
    end

    for (lane, wireindices) in enumerate(wire)
        tn[:plug][(lane, :in)] = wireindices[1]
        tn[:plug][(lane, :out)] = wireindices[2]
    end

    return tn
end

end

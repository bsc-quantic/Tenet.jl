module TenetQuacExt

if isdefined(Base, :get_extension)
    using Tenet
else
    using ..Tenet
end

using Quac: Circuit, lanes, arraytype, Swap
using Bijections

function Tenet.TensorNetwork(circuit::Circuit)
    n = lanes(circuit)
    wire = [[Tenet.letter(i)] for i in 1:n]
    tensors = Tensor[]

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
        push!(tensors, tensor)
    end

    interlayer = [
        Bijection(Dict([site => first(index) for (site, index) in enumerate(wire)])),
        Bijection(Dict([site => last(index) for (site, index) in enumerate(wire)])),
    ]

    return TensorNetwork{Quantum}(tensors; plug = Tenet.Operator, interlayer)
end

end

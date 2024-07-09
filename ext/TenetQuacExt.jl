module TenetQuacExt

using Tenet
using Quac: Gate, Circuit, lanes, arraytype, Swap

function Tenet.Dense(gate::Gate)
    return Tenet.Dense(
        Operator(), arraytype(gate)(gate); sites=Site[Site.(lanes(gate))..., Site.(lanes(gate); dual=true)...]
    )
end

Tenet.evolve!(qtn::Ansatz, gate::Gate; kwargs...) = evolve!(qtn, Tenet.Dense(gate); kwargs...)

function Tenet.Quantum(circuit::Circuit)
    n = lanes(circuit)
    gen = Tenet.IndexCounter()

    wire = [[Tenet.nextindex(gen)] for _ in 1:n]
    tensors = Tensor[]

    for gate in circuit
        G = arraytype(gate)
        array = G(gate)

        if gate isa Swap
            (a, b) = lanes(gate)
            wire[a], wire[b] = wire[b], wire[a]
            continue
        end

        inds = (x -> collect(Iterators.flatten(zip(x...))))(
            map(lanes(gate)) do l
                from, to = last(wire[l]), Tenet.nextindex(gen)
                push!(wire[l], to)
                (from, to)
            end,
        )

        tensor = Tensor(array, tuple(inds...))
        push!(tensors, tensor)
    end

    sites = merge(
        Dict([Site(site; dual=true) => first(index) for (site, index) in enumerate(wire)]),
        Dict([Site(site; dual=false) => last(index) for (site, index) in enumerate(wire)]),
    )

    return Quantum(Tenet.TensorNetwork(tensors), sites)
end

end

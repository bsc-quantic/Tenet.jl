module TenetYaoBlocksExt

using Tenet
using YaoBlocks

function flatten_circuit(x)
    if any(i -> i isa ChainBlock, subblocks(x))
        flatten_circuit(YaoBlocks.Optimise.eliminate_nested(x))
    else
        x
    end
end

function Tenet.Quantum(circuit::AbstractBlock)
    n = nqubits(circuit)
    gen = Tenet.IndexCounter()
    wire = [[Tenet.nextindex!(gen)] for _ in 1:n]
    tensors = Tensor[]

    for gate in flatten_circuit(circuit)
        if gate isa Swap
            (a, b) = occupied_locs(gate)
            wire[a], wire[b] = wire[b], wire[a]
            continue
        end

        # NOTE `YaoBlocks.mat` on m-site qubits still returns the operator on the full Hilbert space
        operator = if gate isa YaoBlocks.ControlBlock
            m = length(occupied_locs(gate))
            control((1:(m - 1))..., m => content(gate))(m)
        else
            content(gate)
        end
        array = reshape(mat(operator), fill(nlevel(operator), 2 * nqubits(operator))...)

        inds = (x -> collect(Iterators.flatten(zip(x...))))(
            map(occupied_locs(gate)) do l
                from, to = last(wire[l]), Tenet.nextindex!(gen)
                push!(wire[l], to)
                (to, from)
            end,
        )

        tensor = Tensor(array, inds)
        push!(tensors, tensor)
    end

    # if a wire has only one index, no gates have been applied to it
    sites = merge(
        Dict([Site(site; dual=true) => first(index) for (site, index) in enumerate(wire) if length(index) > 1]),
        Dict([Site(site; dual=false) => last(index) for (site, index) in enumerate(wire) if length(index) > 1]),
    )

    return Quantum(Tenet.TensorNetwork(tensors), sites)
end

end

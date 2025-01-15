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

function Base.convert(::Type{Circuit}, yaocirc::AbstractBlock)
    circuit = Circuit()

    for gate in flatten_circuit(circuit)
        # if gate isa Swap
        #     (a, b) = occupied_locs(gate)
        #     wire[a], wire[b] = wire[b], wire[a]
        #     continue
        # end

        gatelanes = Lane.(occupied_locs(gate))
        gatesites = [Site.(gatelanes; dual=true)..., Site.(gatelanes)...]

        # NOTE `YaoBlocks.mat` on m-site qubits still returns the operator on the full Hilbert space
        m = length(occupied_locs(gate))
        operator = if gate isa YaoBlocks.ControlBlock
            control((1:(m - 1))..., m => content(gate))(m)
        else
            content(gate)
        end
        array = reshape(collect(mat(operator)), fill(nlevel(operator), length(gatesites))...)

        push!(circuit, Gate(array, gatesites))
    end

    return circuit
end

end

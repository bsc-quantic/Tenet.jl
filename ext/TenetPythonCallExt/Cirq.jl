function Tenet.Quantum(::Val{:cirq}, pyobj::Py)
    cirq = pyimport("cirq")
    if !pyissubclass(pytype(pyobj), cirq.circuits.circuit.Circuit)
        throw(ArgumentError("Expected a cirq.circuits.circuit.Circuit object, got $(pyfullyqualname(pyobj))"))
    end

    gen = Tenet.IndexCounter()

    wire = Dict(qubit => [Tenet.nextindex!(gen)] for qubit in circ.all_qubits())
    tn = TensorNetwork()

    for moment in circ
        for gate in moment.operations
            matrix = pyconvert(Array, cirq.unitary(gate))

            array = reshape(matrix, fill(2, 2 * length(gate.qubits))...)

            inds = (x -> collect(Iterators.flatten(zip(x...))))(
                map(gate.qubits) do l
                    from, to = last(wire[l]), Tenet.nextindex!(gen)
                    push!(wire[l], to)
                    (from, to)
                end,
            )

            tensor = Tensor(array, Tuple(inds))
            push!(tn, tensor)
        end
    end

    function qid_to_site(qid; dual=false)
        if pyissubclass(pytype(qid), cirq.devices.grid_qubit.GridQubit)
            return Site((pyconvert(Int, qid.row), pyconvert(Int, qid.col)); dual)
        else
            throw(ArgumentError("Expected a cirq.devices.grid_qubit.GridQubit object, got $(pyfullyqualname(qid))"))
        end
    end

    sites = merge(
        Dict([qid_to_site(qubit; dual=true) => first(indices) for (qubit, indices) in wire if first(indices) ∈ tn]),
        Dict([qid_to_site(qubit; dual=false) => last(indices) for (qubit, indices) in wire if last(indices) ∈ tn]),
    )

    return Quantum(tn, sites)
end

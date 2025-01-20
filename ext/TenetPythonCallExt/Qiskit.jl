function Base.convert(::Type{Circuit}, ::Val{:qiskit}, pyobj::Py)
    qiskit = pyimport("qiskit")
    if !pyissubclass(pytype(pyobj), qiskit.circuit.quantumcircuit.QuantumCircuit)
        throw(
            ArgumentError(
                "Expected a qiskit.circuit.quantumcircuit.QuantumCircuit object, got $(pyfullyqualname(pyobj))"
            ),
        )
    end

    circuit = Circuit()

    for instr in pyobj
        gatelanes = map(x -> Lane(pyconvert(Int, x._index)), instr.qubits)
        gatesites = [Site.(gatelanes; dual=true)..., Site.(gatelanes)...]

        # if unassigned parameters, throw
        matrix = if pyhasattr(instr, Py("matrix"))
            instr.matrix
        else
            instr.operation.to_matrix()
        end
        if pyisnone(matrix)
            throw(ArgumentError("Expected parameters already assigned, but got $(pyobj.params)"))
        end
        matrix = pyconvert(Array, matrix)
        array = reshape(matrix, fill(2, length(gatesites))...)

        push!(circuit, Gate(array, gatesites))
    end

    return circuit
end

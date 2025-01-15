function Base.convert(::Type{Lane}, ::Val{Symbol("cirq.devices.line_qubit.LineQubit")}, pyobj::Py)
    Lane(pyconvert(Int, pyobj.x))
end

function Base.convert(::Type{Lane}, ::Val{Symbol("cirq.devices.grid_qubit.GridQubit")}, pyobj::Py)
    Lane((pyconvert(Int, pyobj.row), pyconvert(Int, pyobj.col)))
end

function Base.convert(::Type{Circuit}, ::Val{:cirq}, pyobj::Py)
    cirq = pyimport("cirq")
    if !pyissubclass(pytype(pyobj), cirq.circuits.circuit.Circuit)
        throw(ArgumentError("Expected a cirq.circuits.circuit.Circuit object, got $(pyfullyqualname(pyobj))"))
    end

    circuit = Circuit()

    for moment in pyobj
        for gate in moment.operations
            gatelanes = convert.(Lane, gate.qubits)
            gatesites = [Site.(gatelanes; dual=true)..., Site.(gatelanes)...]

            matrix = pyconvert(Array, cirq.unitary(gate))
            array = reshape(matrix, fill(2, length(gatesites))...)

            push!(circuit, Gate(array, gatesites))
        end
    end

    return circuit
end

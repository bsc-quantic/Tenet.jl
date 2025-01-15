using Tenet: Gate

function Base.convert(::Type{Circuit}, ::Val{:qibo}, pyobj::Py)
    qibo = pyimport("qibo")
    qibo.set_backend("numpy")
    if !pyissubclass(pytype(pyobj), qibo.models.circuit.Circuit)
        throw(ArgumentError("Expected a qibo.models.circuit.Circuit object, got $(pyfullyqualname(pyobj))"))
    end

    circuit = Circuit()
    circgates = pyobj.queue

    for gate in circgates
        matrix = pyconvert(Array, gate.matrix())

        gatelanes = map(x -> Lane(pyconvert(Int, x)), gate.qubits)
        gatesites = [Site.(gatelanes; dual=true)..., Site.(gatelanes)...]

        array = reshape(matrix, fill(2, length(gatesites))...)

        push!(circuit, Gate(array, gatesites))
    end

    return circuit
end

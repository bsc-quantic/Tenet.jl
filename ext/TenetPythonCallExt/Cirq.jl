function Base.convert(::Type{Site}, ::Val{Symbol("cirq.devices.line_qubit.LineQubit")}, pyobj::Py)
    Site(pyconvert(Int, pyobj.x))
end

function Base.convert(::Type{Site}, ::Val{Symbol("cirq.devices.grid_qubit.GridQubit")}, pyobj::Py)
    Site((pyconvert(Int, pyobj.row), pyconvert(Int, pyobj.col)))
end

function Base.convert(::Type{Quantum}, ::Val{:cirq}, pyobj::Py)
    cirq = pyimport("cirq")
    if !pyissubclass(pytype(pyobj), cirq.circuits.circuit.Circuit)
        throw(ArgumentError("Expected a cirq.circuits.circuit.Circuit object, got $(pyfullyqualname(pyobj))"))
    end

    gen = Tenet.IndexCounter()

    wire = Dict(qubit => [Tenet.nextindex!(gen)] for qubit in pyobj.all_qubits())
    tn = TensorNetwork()

    for moment in pyobj
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

    sites = merge(
        Dict([convert(Site, qubit)' => first(indices) for (qubit, indices) in wire if first(indices) ∈ tn]),
        Dict([convert(Site, qubit) => last(indices) for (qubit, indices) in wire if last(indices) ∈ tn]),
    )

    return Quantum(tn, sites)
end

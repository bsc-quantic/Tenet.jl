function Tenet.Quantum(::Val{:qibo}, pyobj::Py)
    qibo = pyimport("qibo")
    if !pyissubclass(pytype(pyobj), qibo.models.circuit.Circuit)
        throw(
            ArgumentError(
                "Expected a qibo.models.circuit.Circuit object, got $(pyfullyqualname(pyobj))"
            ),
        )
    end

    n = pyconvert(Int, pyobj.nqubits)
    gen = Tenet.IndexCounter()

    wire = [[Tenet.nextindex!(gen)] for _ in 1:n]
    tn = TensorNetwork()
    circgates = pyobj.queue

    for gate in circgates
        matrix = pyconvert(Array, gate.matrix())

        qubits = map(x -> pyconvert(Int, x), gate.qubits)
        array = reshape(matrix, fill(2, 2 * length(qubits))...)

        inds = (x -> collect(Iterators.flatten(zip(x...))))(
            map(qubits) do l
                l += 1
                from, to = last(wire[l]), Tenet.nextindex!(gen)
                push!(wire[l], to)
                (from, to)
            end,
        )

        tensor = Tensor(array, Tuple(inds))
        push!(tn, tensor)
    end

    sites = merge(
        Dict([Site(site; dual=true) => first(index) for (site, index) in enumerate(wire) if first(index) ∈ tn]),
        Dict([Site(site; dual=false) => last(index) for (site, index) in enumerate(wire) if last(index) ∈ tn]),
    )

    return Quantum(tn, sites)
end
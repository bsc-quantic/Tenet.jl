module TenetPythonCallExt

using Tenet
using PythonCall
using PythonCall.Core: pyisnone

pyfullyqualname(pyobj) = join([pytype(pyobj).__module__, pytype(pyobj).__qualname__], '.')

function Tenet.Quantum(pyobj::Py)
    pyclassname = pyfullyqualname(pyobj)
    if pyclassname != "qiskit.circuit.quantumcircuit.QuantumCircuit"
        throw(ArgumentError("Expected a Qiskit's QuantumCircuit object, got $pyclassname"))
    end

    n = length(pyobj.qregs[0])
    gen = Tenet.IndexCounter()

    wire = [[Tenet.nextindex!(gen)] for _ in 1:n]
    tn = TensorNetwork()

    for instr in pyobj
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

        qubits = map(x -> pyconvert(Int, x._index), instr.qubits)
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

end

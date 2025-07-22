using Tenet
using PythonCall
using QuantumTags

# WARN type-piracy
# TODO move to QuantumTags.jl?
function pyconvert_rule_pytket_qubit(T, pyobj)
    coord = pyconvert(Vector{Int}, pyobj.index)

    # Python uses 0-based indexing, while Julia uses 1-based indexing
    coord .+= 1

    pyconvert_return(T(coord...))
end

function pyconvert_rule_pytket_qubitpaulistring(T, pyobj)
    tn = GenericTensorNetwork()
    for (qubit, pauli) in pyobj.map
        _site = pyconvert(CartesianSite, qubit)

        _array = if pauli.name == "I"
            [1 0; 0 1]
        elseif pauli.name == "X"
            [0 1; 1 0]
        elseif pauli.name == "Y"
            [0 -im; im 0]
        elseif pauli.name == "Z"
            [1 0; 0 -1]
        else
            return pyconvert_unconverted()
        end

        tn[_site] = Tensor(_array, [Index(plug"$_site"), Index(plug"$_site'")])
    end

    pyconvert_return(ProductOperator(tn))
end

function init_pytket()
    pyconvert_add_rule("pytket:Qubit", CartesianSite, pyconvert_rule_pytket_qubit)
    pyconvert_add_rule("pytket.pauli:QubitPauliString", ProductOperator, pyconvert_rule_pytket_qubitpaulistring)
end

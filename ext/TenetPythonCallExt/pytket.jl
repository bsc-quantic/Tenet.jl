using Tenet
using PythonCall
using PythonCall.Convert: pyconvert_add_rule, pyconvert_return, pyconvert_unconverted
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
    for (qubit, pauli) in pyobj.map.items()
        _site = pyconvert(CartesianSite, qubit)
        pauli = pyconvert(String, pauli.name)

        _array = if pauli == "I"
            [1 0; 0 1]
        elseif pauli == "X"
            [0 1; 1 0]
        elseif pauli == "Y"
            [0 -im; im 0]
        elseif pauli == "Z"
            [1 0; 0 -1]
        else
            return pyconvert_unconverted()
        end

        _tensor = Tensor(_array, [Index(plug"$_site"), Index(plug"$_site'")])
        addtensor!(tn, _tensor)
        setsite!(tn, _tensor, _site)
        setplug!(tn, Index(plug"$_site"), plug"$_site")
        setplug!(tn, Index(plug"$_site'"), plug"$_site'")
    end

    pyconvert_return(ProductOperator(tn))
end

function init_pytket()
    pyconvert_add_rule("pytket._tket.unit_id:Qubit", CartesianSite, pyconvert_rule_pytket_qubit)
    pyconvert_add_rule("pytket._tket.pauli:QubitPauliString", ProductOperator, pyconvert_rule_pytket_qubitpaulistring)
end

module TenetPythonCallExt

using Tenet
using PythonCall
using PythonCall.Core: pyisnone

pyfullyqualname(pyobj) = join([pytype(pyobj).__module__, pytype(pyobj).__qualname__], '.')

function Tenet.TensorNetwork(pyobj::Py)
    pymodule, _ = split(pytype(pyobj).__module__, "."; limit=1)
    return TensorNetwork(Val(Symbol(pymodule)), pyobj)
end

function Tenet.Quantum(pyobj::Py)
    pymodule, _ = split(pytype(pyobj).__module__, "."; limit=1)
    return Quantum(Val(Symbol(pymodule)), pyobj)
end

include("Qiskit.jl")
include("Quimb.jl")

end

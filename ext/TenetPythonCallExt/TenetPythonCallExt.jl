module TenetPythonCallExt

using Tenet
using PythonCall
using PythonCall.Core: pyisnone

pyfullyqualname(pyobj) = join([pytype(pyobj).__module__, pytype(pyobj).__qualname__], '.')

Base.convert(::Type{Site}, pyobj::Py) = convert(Site, Val(Symbol(pyfullyqualname(pyobj))), pyobj)

Base.convert(::Type{Lane}, pyobj::Py) = convert(Lane, Val(Symbol(pyfullyqualname(pyobj))), pyobj)

function Base.convert(::Type{TensorNetwork}, pyobj::Py)
    pymodule, _ = split(pyconvert(String, pytype(pyobj).__module__), "."; limit=2)
    return convert(TensorNetwork, Val(Symbol(pymodule)), pyobj)
end

function Base.convert(::Type{Quantum}, pyobj::Py)
    pymodule, _ = split(pyconvert(String, pytype(pyobj).__module__), "."; limit=2)
    return convert(Quantum, Val(Symbol(pymodule)), pyobj)
end

function Base.convert(::Type{Circuit}, pyobj::Py)
    pymodule, _ = split(pyconvert(String, pytype(pyobj).__module__), "."; limit=2)
    return convert(Circuit, Val(Symbol(pymodule)), pyobj)
end

include("Qiskit.jl")
include("Quimb.jl")
include("Qibo.jl")
include("Cirq.jl")

end

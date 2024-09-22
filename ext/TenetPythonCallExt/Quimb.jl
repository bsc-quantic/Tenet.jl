function Tenet.TensorNetwork(::Val{:quimb}, pyobj::Py)
    quimb = pyimport("quimb")
    if !pyissubclass(pytype(pyobj), quimb.tensor.tensor_core.TensorNetwork)
        throw(ArgumentError("Expected a quimb.tensor.tensor_core.TensorNetwork object, got $(pyfullyqualname(pyobj))"))
    end

    ts = map(pyobj.tensors) do tensor
        array = pyconvert(Array, tensor.data)
        inds = Symbol.(pyconvert(Array, tensor.inds))
        Tensor(array, inds)
    end

    return TensorNetwork(ts)
end

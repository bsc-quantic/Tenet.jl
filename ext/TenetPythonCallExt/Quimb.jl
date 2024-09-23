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

function Tenet.Quantum(::Val{:quimb}, pyobj::Py)
    quimb = pyimport("quimb")
    if pyissubclass(pytype(pyobj), quimb.tensor.circuit.Circuit)
        return Quantum(pyobj.get_uni())
    elseif pyissubclass(pytype(pyobj), quimb.tensor.tensor_arbgeom.TensorNetworkGenVector)
        return Quantum(Val(Symbol("quimb.tensor.tensor_arbgeom.TensorNetworkGenVector")), pyobj)
    elseif pyissubclass(pytype(pyobj), quimb.tensor.tensor_arbgeom.TensorNetworkGenOperator)
        return Quantum(Val(Symbol("quimb.tensor.tensor_arbgeom.TensorNetworkGenOperator")), pyobj)
    else
        throw(ArgumentError("Unknown treatment for object of class $(pyfullyqualname(pyobj))"))
    end
end

function Tenet.Quantum(::Val{Symbol("quimb.tensor.tensor_arbgeom.TensorNetworkGenVector")}, pyobj::Py)
    tn = TensorNetwork(pyobj)
    sitedict = Dict(Site(pyconvert(Int, i)) => pyconvert(Symbol, pyobj.site_ind(i)) for i in pyobj.sites)
    return Quantum(tn, sitedict)
end

function Tenet.Quantum(::Val{Symbol("quimb.tensor.tensor_arbgeom.TensorNetworkGenOperator")}, pyobj::Py)
    tn = TensorNetwork(pyobj)

    sitedict = merge!(
        Dict(Site(pyconvert(Int, i)) => pyconvert(Symbol, pyobj.lower_ind(i)) for i in pyobj.sites),
        Dict(Site(pyconvert(Int, i); dual=true) => pyconvert(Symbol, pyobj.upper_ind(i)) for i in pyobj.sites),
    )

    return Quantum(tn, sitedict)
end

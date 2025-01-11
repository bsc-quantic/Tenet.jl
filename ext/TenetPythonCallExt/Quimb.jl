function Base.convert(::Type{TensorNetwork}, ::Val{:quimb}, pyobj::Py)
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

function Base.convert(::Type{Quantum}, ::Val{:quimb}, pyobj::Py)
    quimb = pyimport("quimb")
    if pyissubclass(pytype(pyobj), quimb.tensor.circuit.Circuit)
        return convert(Quantum, pyobj.get_uni())
    elseif pyissubclass(pytype(pyobj), quimb.tensor.tensor_arbgeom.TensorNetworkGenVector)
        return convert(Quantum, Val(Symbol("quimb.tensor.tensor_arbgeom.TensorNetworkGenVector")), pyobj)
    elseif pyissubclass(pytype(pyobj), quimb.tensor.tensor_arbgeom.TensorNetworkGenOperator)
        return convert(Quantum, Val(Symbol("quimb.tensor.tensor_arbgeom.TensorNetworkGenOperator")), pyobj)
    else
        throw(ArgumentError("Unknown treatment for object of class $(pyfullyqualname(pyobj))"))
    end
end

function Base.convert(::Type{Quantum}, ::Val{Symbol("quimb.tensor.tensor_arbgeom.TensorNetworkGenVector")}, pyobj::Py)
    tn = convert(TensorNetwork, pyobj)
    sitedict = Dict(Site(pyconvert(Int, i)) => pyconvert(Symbol, pyobj.site_ind(i)) for i in pyobj.sites)
    return Quantum(tn, sitedict)
end

function Base.convert(::Type{Quantum}, ::Val{Symbol("quimb.tensor.tensor_arbgeom.TensorNetworkGenOperator")}, pyobj::Py)
    tn = convert(TensorNetwork, pyobj)

    sitedict = merge!(
        Dict(Site(pyconvert(Int, i)) => pyconvert(Symbol, pyobj.lower_ind(i)) for i in pyobj.sites),
        Dict(Site(pyconvert(Int, i); dual=true) => pyconvert(Symbol, pyobj.upper_ind(i)) for i in pyobj.sites),
    )

    return Quantum(tn, sitedict)
end

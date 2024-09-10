module TenetITensorsExt

using Tenet
using ITensors: ITensors, ITensor

function Tenet.Tensor(tensor::ITensor)
    array = ITensors.array(tensor)
    is = Symbol.(ITensors.id.(ITensors.inds(tensor)))
    return Tensor(array, is)
end

Tenet.TensorNetwork(tn::Vector{ITensor}) = TensorNetwork(map(Tensor, tn))

end

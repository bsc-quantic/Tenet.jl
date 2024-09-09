module TenetITensorsExt

using Tenet
using ITensors: ITensors, ITensor

function Tenet.Tensor(tensor::ITensor)
    array = ITensors.array(tensor)
    is = Symbol.(id.(ITensor.inds(tensor)))
    return Tensor(array, is)
end

end

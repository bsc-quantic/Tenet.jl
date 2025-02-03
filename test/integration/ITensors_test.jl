using Test
using Tenet
using ITensors: ITensors, ITensor, Index, array

i = Index(2, "i")
j = Index(3, "j")
k = Index(4, "k")

itensor = ITensor(rand(2, 3, 4), i, j, k)

tensor = convert(Tensor, itensor)
@test tensor isa Tensor
@test size(tensor) == (2, 3, 4)
@test parent(tensor) == array(itensor)

tensor = Tensor(rand(2, 3, 4), (:i, :j, :k))
itensor = convert(ITensor, tensor)
@test itensor isa ITensor
@test size(itensor) == (2, 3, 4)
@test array(itensor) == parent(tensor)
@test all(
    splat(==), zip(map(x -> replace(x, "\"" => ""), string.(ITensors.tags.(ITensors.inds(itensor)))), ["i", "j", "k"])
)

tn = rand(TensorNetwork, 4, 3)
itensors = convert(Vector{ITensor}, tn)
@test itensors isa Vector{ITensor}

tnr = convert(TensorNetwork, itensors)
@test tnr isa TensorNetwork
@test issetequal(arrays(tn), arrays(tnr))

using Test
using Tenet: Tenet, MPS, form, Site, Lane, MixedCanonical
using ITensorMPS
using ITensors
using ITensors: ITensor, Index, dim, dims

# Tenet to ITensorMPS conversion
tenet_mps = rand(MPS; n=5, maxdim=30)
itensor_mps = convert(ITensorMPS.MPS, tenet_mps)

@test length(Tenet.tensors(tenet_mps)) == length(ITensors.tensors(itensor_mps))

for (t1, t2) in zip(Tenet.tensors(tenet_mps), ITensors.tensors(itensor_mps))
    @test issetequal(size(t1), dims(t2))
end

@test itensor_mps.llim == Tenet.id(form(tenet_mps).orthog_center) - 1
@test itensor_mps.rlim == Tenet.id(form(tenet_mps).orthog_center) + 1

contracted = Tenet.contract(tenet_mps)
permuted = permutedims(contracted, [Tenet.inds(tenet_mps; at=Site(i)) for i in 1:length(Tenet.tensors(tenet_mps))])
@test isapprox(parent(permuted), Array(ITensorMPS.contract(itensor_mps).tensor))

# ITensorMPS to Tenet conversion
itensor_mps = ITensorMPS.random_mps(siteinds(4, 5); linkdims=7)
tenet_mps = convert(MPS, itensor_mps)

@test length(ITensors.tensors(itensor_mps)) == length(Tenet.tensors(tenet_mps))

for (t1, t2) in zip(ITensors.tensors(itensor_mps), Tenet.tensors(tenet_mps))
    @test issetequal(dims(t1), size(t2))
end

@test form(tenet_mps) isa MixedCanonical
@test form(tenet_mps).orthog_center == Lane(itensor_mps.llim + 1)
@test form(tenet_mps).orthog_center == Lane(itensor_mps.rlim - 1)

contracted = Tenet.contract(tenet_mps)
permuted = permutedims(contracted, [Tenet.inds(tenet_mps; at=Site(i)) for i in 1:length(Tenet.tensors(tenet_mps))])
@test isapprox(parent(permuted), Array(ITensorMPS.contract(itensor_mps).tensor))

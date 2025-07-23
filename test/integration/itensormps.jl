using Test
using Tenet: Tenet, @site_str, @plug_str, @bond_str
using ITensors
using ITensors: ITensor, Index, dim, dims
using ITensorMPS

# TODO update code when conversion from Tenet to ITensorMPS is reimplemented
# Tenet to ITensorMPS conversion
# tenet_mps = rand(Tenet.MPS; n=5, maxdim=30)
# itensor_mps = convert(ITensorMPS.MPS, tenet_mps)

# @test length(Tenet.tensors(tenet_mps)) == length(ITensorMPS.tensors(itensor_mps))

# for (t1, t2) in zip(Tenet.tensors(tenet_mps), ITensorMPS.tensors(itensor_mps))
#     @test issetequal(size(t1), dims(t2))
# end

# @test itensor_mps.llim == Tenet.id(form(tenet_mps).orthog_center) - 1
# @test itensor_mps.rlim == Tenet.id(form(tenet_mps).orthog_center) + 1

# contracted = Tenet.contract(tenet_mps)
# permuted = permutedims(contracted, [Tenet.inds(tenet_mps; at=Site(i)) for i in 1:length(Tenet.tensors(tenet_mps))])
# @test isapprox(parent(permuted), Array(ITensorMPS.contract(itensor_mps).tensor))

# ITensorMPS to Tenet conversion
itensor_mps = ITensorMPS.random_mps(siteinds(4, 5); linkdims=7)
tenet_mps = convert(Tenet.MPS, itensor_mps)

@test length(ITensorMPS.tensors(itensor_mps)) == Tenet.ntensors(tenet_mps)

for (i, _siteind) in enumerate(siteinds(itensor_mps))
    plugind = ind_at(tenet_mps, plug"$i")
    @test size(tenet_mps, plugind) == dim(_siteind)
end

for (i, _linkind) in enumerate(linkinds(itensor_mps))
    bondind = ind_at(tenet_mps, bond"$i-$(i+1)")
    @test size(tenet_mps, bondind) == dim(_linkind)
end

@test Tenet.form(tenet_mps) isa Tenet.MixedCanonical
@test Tenet.max_orthog_center(Tenet.form(tenet_mps)) == site"$(itensor_mps.llim + 1)"
@test Tenet.min_orthog_center(Tenet.form(tenet_mps)) == site"$(itensor_mps.rlim - 1)"

tenet_contracted = Tenet.contract(tenet_mps)
tenet_contracted = parent(
    permutedims(tenet_contracted, [Tenet.ind_at(tenet_mps, plug"$i") for i in 1:Tenet.ntensors(tenet_mps)])
)

itensor_contracted = Array(ITensorMPS.contract(itensor_mps).tensor)
@test isapprox(tenet_contracted, itensor_contracted)

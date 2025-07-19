using Test
using Tenet
using Tenet: checkform, min_orthog_center, max_orthog_center
using Muscle: isisometry

@testset "case 1" begin
    a = ones(2, 3, 4)
    b = 2ones(2, 2, 3, 4)
    c = 3ones(2, 3, 4)

    H = MPO([a, b, c])

    @test issetequal(sites(H), [site"1", site"2", site"3"])
    @test issetequal(bonds(H), [bond"1-2", bond"2-3"])
    @test issetequal(plugs(H), [plug"1", plug"2", plug"3", plug"1'", plug"2'", plug"3'"])

    @test parent(H[site"1"]) == a
    @test parent(H[site"2"]) == b
    @test parent(H[site"3"]) == c

    @test size(H, H[plug"1"]) == 3
    @test size(H, H[plug"2"]) == 3
    @test size(H, H[plug"3"]) == 3
    @test size(H, H[plug"1'"]) == 4
    @test size(H, H[plug"2'"]) == 4
    @test size(H, H[plug"3'"]) == 4
    @test size(H, H[bond"1-2"]) == 2
    @test size(H, H[bond"2-3"]) == 2
end

@testset "case 2" begin
    a = ones(4, 7, 1)
    b = 2ones(4, 5, 6, 3)
    c = 3ones(5, 8, 2)

    H = MPO([a, b, c])

    @test issetequal(sites(H), [site"1", site"2", site"3"])
    @test issetequal(bonds(H), [bond"1-2", bond"2-3"])
    @test issetequal(plugs(H), [plug"1", plug"2", plug"3", plug"1'", plug"2'", plug"3'"])

    @test parent(H[site"1"]) == a
    @test parent(H[site"2"]) == b
    @test parent(H[site"3"]) == c

    @test size(H, H[plug"1"]) == 7
    @test size(H, H[plug"2"]) == 6
    @test size(H, H[plug"3"]) == 8
    @test size(H, H[plug"1'"]) == 1
    @test size(H, H[plug"2'"]) == 3
    @test size(H, H[plug"3'"]) == 2
    @test size(H, H[bond"1-2"]) == 4
    @test size(H, H[bond"2-3"]) == 5
end

@testset "case 3: order = [:r, :o, :l, :i]" begin
    a = ones(4, 7, 1)
    b = 2ones(5, 6, 4, 3)
    c = 3ones(8, 5, 2)

    H = MPO([a, b, c]; order=[:r, :o, :l, :i])

    @test issetequal(sites(H), [site"1", site"2", site"3"])
    @test issetequal(bonds(H), [bond"1-2", bond"2-3"])
    @test issetequal(plugs(H), [plug"1", plug"2", plug"3", plug"1'", plug"2'", plug"3'"])

    @test parent(H[site"1"]) == a
    @test parent(H[site"2"]) == b
    @test parent(H[site"3"]) == c

    @test size(H, H[plug"1"]) == 7
    @test size(H, H[plug"2"]) == 6
    @test size(H, H[plug"3"]) == 8
    @test size(H, H[plug"1'"]) == 1
    @test size(H, H[plug"2'"]) == 3
    @test size(H, H[plug"3'"]) == 2
    @test size(H, H[bond"1-2"]) == 4
    @test size(H, H[bond"2-3"]) == 5
end

@testset "canonize! > MPO > MixedCanonical > multiple Sites" begin
    ψ = MPO([
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
    ])
    ψc = canonize(ψ, [site"2", site"3"])

    # TODO implement `checkform` for MPO on `MixedCanonical`
    @test checkform(ψc) skip = true
    @test form(ψc) == MixedCanonical([site"2", site"3"])
    @test min_orthog_center(form(ψc)) == site"2"
    @test max_orthog_center(form(ψc)) == site"3"

    @test isisometry(ψc[site"1"], Index(bond"1-2"))
    @test !isisometry(ψc[site"2"], Index(bond"1-2"))
    @test !isisometry(ψc[site"3"], Index(bond"3-4"))
    @test isisometry(ψc[site"4"], Index(bond"3-4"))
    @test isisometry(ψc[site"5"], Index(bond"4-5"))

    @test contract(ψc) ≈ contract(ψ)
end

@testset "canonize! > MPO > MixedCanonical > single site" begin
    ψ = MPO([
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
    ])
    ψc = canonize(ψ, site"3")

    # TODO implement `checkform` for MPO on `MixedCanonical`
    @test checkform(ψc) skip = true
    @test form(ψc) == MixedCanonical(site"3")
    @test min_orthog_center(form(ψc)) == site"3"
    @test max_orthog_center(form(ψc)) == site"3"

    @test isisometry(ψc[site"1"], Index(bond"1-2"))
    @test isisometry(ψc[site"2"], Index(bond"2-3"))
    @test !isisometry(ψc[site"3"], Index(bond"2-3"))
    @test !isisometry(ψc[site"3"], Index(bond"3-4"))
    @test isisometry(ψc[site"4"], Index(bond"3-4"))
    @test isisometry(ψc[site"5"], Index(bond"4-5"))

    @test contract(ψc) ≈ contract(ψ)
end

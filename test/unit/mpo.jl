using Test
using Tenet
using Tenet: checkform, orthog_center, min_orthog_center, max_orthog_center, unsafe_setform!, LambdaSite
using Muscle: hadamard, isisometry

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

@testset "canonize!: NonCanonical to MixedCanonical single site" begin
    ψ = MPO([
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
    ])
    unsafe_setform!(ψ, NonCanonical())
    ψc = canonize(ψ, site"3")

    @test ntensors(ψc) == 5
    @test issetequal(all_sites(ψc), [site"1", site"2", site"3", site"4", site"5"])
    @test issetequal(all_bonds(ψc), [bond"1-2", bond"2-3", bond"3-4", bond"4-5"])

    @test checkform(ψc)
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

@testset "canonize!: NonCanonical to MixedCanonical multiple sites" begin
    ψ = MPO([
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
    ])
    unsafe_setform!(ψ, NonCanonical())
    ψc = canonize(ψ, [site"2", site"3"])

    @test ntensors(ψc) == 5
    @test issetequal(all_sites(ψc), [site"1", site"2", site"3", site"4", site"5"])
    @test issetequal(all_bonds(ψc), [bond"1-2", bond"2-3", bond"3-4", bond"4-5"])

    @test checkform(ψc)
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

@testset "canonize!: MixedCanonical single site to MixedCanonical single site (sweep)" begin
    ψ = MPO([
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
    ])
    canonize!(ψ, site"5")
    ψc = canonize(ψ, site"3")

    @test ntensors(ψc) == 5
    @test issetequal(all_sites(ψc), [site"1", site"2", site"3", site"4", site"5"])
    @test issetequal(all_bonds(ψc), [bond"1-2", bond"2-3", bond"3-4", bond"4-5"])

    @test checkform(ψc)
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

@testset "canonize!: MixedCanonical multiple sites to MixedCanonical single site" begin
    ψ = MPO([
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
    ])
    unsafe_setform!(ψ, MixedCanonical(all_sites(ψ)))
    ψc = canonize(ψ, site"3")

    @test ntensors(ψc) == 5
    @test issetequal(all_sites(ψc), [site"1", site"2", site"3", site"4", site"5"])
    @test issetequal(all_bonds(ψc), [bond"1-2", bond"2-3", bond"3-4", bond"4-5"])

    @test checkform(ψc)
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

@testset "canonize!: MixedCanonical multiple sites to MixedCanonical multiple sites (smaller range)" begin
    ψ = MPO([
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
    ])
    unsafe_setform!(ψ, MixedCanonical(all_sites(ψ)))
    ψc = canonize(ψ, [site"2", site"3"])

    @test ntensors(ψc) == 5
    @test issetequal(all_sites(ψc), [site"1", site"2", site"3", site"4", site"5"])
    @test issetequal(all_bonds(ψc), [bond"1-2", bond"2-3", bond"3-4", bond"4-5"])

    @test checkform(ψc)
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

@testset "canonize!: NonCanonical to BondCanonical" begin
    ψ = MPO([
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
    ])
    unsafe_setform!(ψ, NonCanonical())
    ψc = canonize(ψ, BondCanonical(bond"2-3"))

    @test ntensors(ψc) == 6
    @test issetequal(all_sites(ψc), [site"1", site"2", site"3", site"4", site"5", LambdaSite(bond"2-3")])
    @test issetequal(all_bonds(ψc), [bond"1-2", bond"2-3", bond"3-4", bond"4-5"])

    @test checkform(ψc)
    @test form(ψc) == BondCanonical(bond"2-3")
    @test orthog_center(form(ψc)) == bond"2-3"

    @test isisometry(ψc[site"1"], Index(bond"1-2"))
    @test !isisometry(ψc[site"2"], Index(bond"1-2"))
    @test !isisometry(ψc[site"3"], Index(bond"3-4"))
    @test isisometry(ψc[site"4"], Index(bond"3-4"))
    @test isisometry(ψc[site"5"], Index(bond"4-5"))

    # TODO fix easy Hadamard products on `contract`
    s = ψc[LambdaSite(bond"2-3")]
    ψc[site"2"] = hadamard(ψc[site"2"], s)
    rmtensor!(ψc, s)

    @test contract(ψc) ≈ contract(ψ)
end

@testset "canonize!: MixedCanonical single site to BondCanonical" begin
    ψ = MPO([
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
    ])
    canonize!(ψ, MixedCanonical(site"5"))
    ψc = canonize(ψ, BondCanonical(bond"2-3"))

    @test ntensors(ψc) == 6
    @test issetequal(all_sites(ψc), [site"1", site"2", site"3", site"4", site"5", LambdaSite(bond"2-3")])
    @test issetequal(all_bonds(ψc), [bond"1-2", bond"2-3", bond"3-4", bond"4-5"])

    @test checkform(ψc)
    @test form(ψc) == BondCanonical(bond"2-3")

    @test isisometry(ψc[site"1"], Index(bond"1-2"))
    @test !isisometry(ψc[site"2"], Index(bond"1-2"))
    @test !isisometry(ψc[site"3"], Index(bond"3-4"))
    @test isisometry(ψc[site"4"], Index(bond"3-4"))
    @test isisometry(ψc[site"5"], Index(bond"4-5"))

    # TODO fix easy Hadamard products on `contract`
    s = ψc[LambdaSite(bond"2-3")]
    ψc[site"2"] = hadamard(ψc[site"2"], s)
    rmtensor!(ψc, s)

    @test contract(ψc) ≈ contract(ψ)
end

@testset "canonize!: MixedCanonical multiple sites to BondCanonical (no overlap)" begin
    ψ = MPO([
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
    ])
    canonize!(ψ, MixedCanonical([site"3", site"4", site"5"]))
    ψc = canonize(ψ, BondCanonical(bond"1-2"))

    @test ntensors(ψc) == 6
    @test issetequal(all_sites(ψc), [site"1", site"2", site"3", site"4", site"5", LambdaSite(bond"1-2")])
    @test issetequal(all_bonds(ψc), [bond"1-2", bond"2-3", bond"3-4", bond"4-5"])

    @test checkform(ψc)
    @test form(ψc) == BondCanonical(bond"1-2")

    @test !isisometry(ψc[site"2"], Index(bond"2-3"))
    @test isisometry(ψc[site"3"], Index(bond"2-3"))
    @test isisometry(ψc[site"4"], Index(bond"3-4"))
    @test isisometry(ψc[site"5"], Index(bond"4-5"))

    # TODO fix easy Hadamard products on `contract`
    s = ψc[LambdaSite(bond"1-2")]
    ψc[site"2"] = hadamard(ψc[site"2"], s)
    rmtensor!(ψc, s)

    @test contract(ψc) ≈ contract(ψ)
end

@testset "canonize!: MixedCanonical multiple sites to BondCanonical (overlap)" begin
    ψ = MPO([
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
    ])
    canonize!(ψ, MixedCanonical([site"3", site"4", site"5"]))
    ψc = canonize(ψ, BondCanonical(bond"2-3"))

    @test ntensors(ψc) == 6
    @test issetequal(all_sites(ψc), [site"1", site"2", site"3", site"4", site"5", LambdaSite(bond"2-3")])
    @test issetequal(all_bonds(ψc), [bond"1-2", bond"2-3", bond"3-4", bond"4-5"])

    @test checkform(ψc)
    @test form(ψc) == BondCanonical(bond"2-3")

    @test isisometry(ψc[site"1"], Index(bond"1-2"))
    @test !isisometry(ψc[site"2"], Index(bond"1-2"))
    @test !isisometry(ψc[site"3"], Index(bond"3-4"))
    @test isisometry(ψc[site"4"], Index(bond"3-4"))
    @test isisometry(ψc[site"5"], Index(bond"4-5"))

    # TODO fix easy Hadamard products on `contract`
    s = ψc[LambdaSite(bond"2-3")]
    ψc[site"2"] = hadamard(ψc[site"2"], s)
    rmtensor!(ψc, s)

    @test contract(ψc) ≈ contract(ψ)
end

@testset "canonize!: BondCanonical to BondCanonical" begin
    ψ = MPO([
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
    ])
    canonize!(ψ, BondCanonical(bond"4-5"))
    ψc = canonize(ψ, BondCanonical(bond"2-3"))

    @test ntensors(ψc) == 6
    @test issetequal(all_sites(ψc), [site"1", site"2", site"3", site"4", site"5", LambdaSite(bond"2-3")])
    @test issetequal(all_bonds(ψc), [bond"1-2", bond"2-3", bond"3-4", bond"4-5"])

    @test checkform(ψc)
    @test form(ψc) == BondCanonical(bond"2-3")

    @test isisometry(ψc[site"1"], Index(bond"1-2"))
    @test !isisometry(ψc[site"2"], Index(bond"1-2"))
    @test !isisometry(ψc[site"3"], Index(bond"3-4"))
    @test isisometry(ψc[site"4"], Index(bond"3-4"))
    @test isisometry(ψc[site"5"], Index(bond"4-5"))

    # TODO fix easy Hadamard products on `contract`
    s = ψc[LambdaSite(bond"2-3")]
    ψc[site"2"] = hadamard(ψc[site"2"], s)
    rmtensor!(ψc, s)

    s = ψ[LambdaSite(bond"4-5")]
    ψ[site"5"] = hadamard(ψ[site"5"], s)
    rmtensor!(ψ, s)

    # atol because there were some weird numerical issues
    @test contract(ψc) ≈ contract(ψ)
end

@testset "canonize!: BondCanonical to MixedCanonical single site (overlap)" begin
    ψ = MPO([
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
    ])
    canonize!(ψ, BondCanonical(bond"2-3"))
    ψc = canonize(ψ, site"3")

    @test ntensors(ψc) == 5
    @test issetequal(all_sites(ψc), [site"1", site"2", site"3", site"4", site"5"])
    @test issetequal(all_bonds(ψc), [bond"1-2", bond"2-3", bond"3-4", bond"4-5"])

    @test checkform(ψc)
    @test form(ψc) == MixedCanonical(site"3")
    @test min_orthog_center(form(ψc)) == site"3"
    @test max_orthog_center(form(ψc)) == site"3"

    @test isisometry(ψc[site"1"], Index(bond"1-2"))
    @test isisometry(ψc[site"2"], Index(bond"2-3"))
    @test !isisometry(ψc[site"3"], Index(bond"2-3"))
    @test !isisometry(ψc[site"3"], Index(bond"3-4"))
    @test isisometry(ψc[site"4"], Index(bond"3-4"))
    @test isisometry(ψc[site"5"], Index(bond"4-5"))

    # TODO fix easy Hadamard products on `contract`
    s = ψ[LambdaSite(bond"2-3")]
    ψ[site"2"] = hadamard(ψ[site"2"], s)
    rmtensor!(ψ, s)

    @test contract(ψc) ≈ contract(ψ)
end

@testset "canonize!: BondCanonical to MixedCanonical single site (no overlap)" begin
    ψ = MPO([
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:256, 4, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
    ])
    canonize!(ψ, BondCanonical(bond"4-5"))
    ψc = canonize(ψ, site"3")

    @test ntensors(ψc) == 5
    @test issetequal(all_sites(ψc), [site"1", site"2", site"3", site"4", site"5"])
    @test issetequal(all_bonds(ψc), [bond"1-2", bond"2-3", bond"3-4", bond"4-5"])

    @test checkform(ψc)
    @test form(ψc) == MixedCanonical(site"3")
    @test min_orthog_center(form(ψc)) == site"3"
    @test max_orthog_center(form(ψc)) == site"3"

    @test isisometry(ψc[site"1"], Index(bond"1-2"))
    @test isisometry(ψc[site"2"], Index(bond"2-3"))
    @test !isisometry(ψc[site"3"], Index(bond"2-3"))
    @test !isisometry(ψc[site"3"], Index(bond"3-4"))
    @test isisometry(ψc[site"4"], Index(bond"3-4"))
    @test isisometry(ψc[site"5"], Index(bond"4-5"))

    # TODO fix easy Hadamard products on `contract`
    s = ψ[LambdaSite(bond"4-5")]
    ψ[site"4"] = hadamard(ψ[site"4"], s)
    rmtensor!(ψ, s)

    @test contract(ψc) ≈ contract(ψ)
end

@testset "compress!: single bond" begin
    ψ = MPS([
        ones(Int, 4, 4), # the rank is 1 so compress up to size 1 should not change it
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:16, 4, 4)),
    ])

    @testset let ψc = compress(ψ, bond"1-2"; maxdim=1)
        @test size(ψc, ψc[bond"1-2"]) == 1

        # TODO fix easy Hadamard products on `contract`
        s = ψc[LambdaSite(bond"1-2")]
        ψc[site"1"] = hadamard(ψc[site"1"], s)
        rmtensor!(ψc, s)

        @test contract(ψc) ≈ contract(ψ)
    end

    @testset let ψc = compress(ψ, bond"1-2"; threshold=0.1)
        @test size(ψc, ψc[bond"1-2"]) == 1

        # TODO fix easy Hadamard products on `contract`
        s = ψc[LambdaSite(bond"1-2")]
        ψc[site"1"] = hadamard(ψc[site"1"], s)
        rmtensor!(ψc, s)

        @test contract(ψc) ≈ contract(ψ)
    end
end

@testset "compress!: all bonds" begin
    ψ = MPS([
        ones(Int, 4, 4), # the rank is 1 so compress up to size 1 should not change it
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:16, 4, 4)),
    ])

    @testset let ψc = compress(ψ; maxdim=1)
        @test all(b -> size(ψc, ψc[b]) == 1, all_bonds(ψc))
    end

    # use 2nd Schmidt value increased by 10% as threshold
    threshold = Tenet.schmidt_values(ψ, bond"1-2")[2] * 1.1
    @testset let ψc = compress(ψ; threshold)
        @test size(ψc, ψc[bond"1-2"]) == 1

        # TODO fix easy Hadamard products on `contract`
        s = ψc[LambdaSite(bond"4-5")]
        ψc[site"5"] = hadamard(ψc[site"5"], s)
        rmtensor!(ψc, s)

        @test contract(ψc) ≈ contract(ψ)
    end
end

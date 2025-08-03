using Test
using Tenet
using Tenet: checkform, orthog_center, min_orthog_center, max_orthog_center, unsafe_setform!, LambdaSite
using Muscle: hadamard, isisometry

@testset "constructor: case 1" begin
    a = ones(1, 2)
    b = 2ones(1, 3, 4)
    c = 3ones(3, 5)

    tn = MPS([a, b, c])

    @test issetequal(sites(tn), [site"1", site"2", site"3"])
    @test issetequal(bonds(tn), [bond"1-2", bond"2-3"])
    @test issetequal(plugs(tn), [plug"1", plug"2", plug"3"])

    @test parent(tn[site"1"]) == a
    @test parent(tn[site"2"]) == b
    @test parent(tn[site"3"]) == c

    @test size(tn, tn[plug"1"]) == 2
    @test size(tn, tn[plug"2"]) == 4
    @test size(tn, tn[plug"3"]) == 5
    @test size(tn, tn[bond"1-2"]) == 1
    @test size(tn, tn[bond"2-3"]) == 3
end

@testset "constructor: case 2: order = [:r, :o, :l]" begin
    a = ones(1, 2)
    b = 2ones(3, 2, 1)
    c = 3ones(2, 3)

    tn = MPS([a, b, c]; order=[:r, :o, :l])

    @test issetequal(sites(tn), [site"1", site"2", site"3"])
    @test issetequal(bonds(tn), [bond"1-2", bond"2-3"])
    @test issetequal(plugs(tn), [plug"1", plug"2", plug"3"])

    @test parent(tn[site"1"]) == a
    @test parent(tn[site"2"]) == b
    @test parent(tn[site"3"]) == c

    @test size(tn, tn[plug"1"]) == 2
    @test size(tn, tn[plug"2"]) == 2
    @test size(tn, tn[plug"3"]) == 2
    @test size(tn, tn[bond"1-2"]) == 1
    @test size(tn, tn[bond"2-3"]) == 3
end

@testset "conversion: ProductState -> MPS" begin
    a = ProductState([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0] / sqrt(2)])
    b = convert(MPS, a)

    @test issetequal(sites(b), [site"1", site"2", site"3"])
    @test issetequal(bonds(b), [bond"1-2", bond"2-3"])
    @test issetequal(plugs(b), [plug"1", plug"2", plug"3"])
    @test vec(parent(b[site"1"])) == [1.0, 0.0]
    @test vec(parent(b[site"2"])) == [0.0, 1.0]
    @test vec(parent(b[site"3"])) == [1.0, 1.0] / sqrt(2)
end

@testset "canonize!: NonCanonical to MixedCanonical single site" begin
    ψ = MPS([
        collect(reshape(1:16, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:16, 4, 4)),
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
    ψ = MPS([
        collect(reshape(1:16, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:16, 4, 4)),
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
    ψ = MPS([
        collect(reshape(1:16, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:16, 4, 4)),
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
    ψ = MPS([
        collect(reshape(1:16, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:16, 4, 4)),
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
    ψ = MPS([
        collect(reshape(1:16, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:16, 4, 4)),
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
    ψ = MPS([
        collect(reshape(1:16, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:16, 4, 4)),
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
    ψ = MPS([
        collect(reshape(1:16, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:16, 4, 4)),
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
    ψ = MPS([
        collect(reshape(1:16, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:16, 4, 4)),
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
    ψ = MPS([
        collect(reshape(1:16, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:16, 4, 4)),
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
    ψ = MPS([
        collect(reshape(1:16, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:16, 4, 4)),
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
    ψ = MPS([
        collect(reshape(1:16, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:16, 4, 4)),
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
    ψ = MPS([
        collect(reshape(1:16, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:64, 4, 4, 4)),
        collect(reshape(1:16, 4, 4)),
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

@testset "sample" begin
    # uses a boolean MPS for testing:
    # encodes boolean formula `(a₁ ∧ a₂ ∧ ¬a₃ ∧ a₄) ∨ (¬a₁ ∧ ¬a₂ ∧ a₃ ∧ ¬a₄)`
    # which represents state `|0010> + |1101>`
    tn = MPS([zeros(2, 2), zeros(2, 2, 2), zeros(2, 2, 2), zeros(2, 2)])

    tn[site"1"] .= [1 0; 0 1]

    view(tn[site"2"], [Index(bond"1-2") => 1, Index(plug"2") => 1, Index(bond"2-3") => 1]...) .= 1
    view(tn[site"2"], [Index(bond"1-2") => 2, Index(plug"2") => 2, Index(bond"2-3") => 2]...) .= 1

    view(tn[site"3"], [Index(bond"2-3") => 1, Index(plug"3") => 2, Index(bond"3-4") => 2]...) .= 1
    view(tn[site"3"], [Index(bond"2-3") => 2, Index(plug"3") => 1, Index(bond"3-4") => 1]...) .= 1

    tn[site"4"] .= [0 1; 1 0] / sqrt(2)

    samples = Tenet.sample(tn, 128; batchdim=4)
    @test all(∈(([1, 1, 2, 1], [2, 2, 1, 2])), samples)
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

@testset "bondsizes" begin
    psi = rand(MPS; n=8)
    @test Tenet.bondsizes(psi; sorted=true) == [2, 4, 8, 16, 8, 4, 2]
    @test Tenet.maxbondsize(psi) == 16

    Tenet.truncate!(psi, maxdim=10)
    @test Tenet.maxbondsize(psi) == 10
end

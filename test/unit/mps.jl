using Test
using Tenet

@testset "case 1" begin
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

@testset "case 2: order = [:r, :o, :l]" begin
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

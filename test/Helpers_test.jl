@testset "Helpers" begin
    @testset "RingPeek" begin
        using Tenet: ringpeek

        it = ringpeek([0 1 2])
        @test length(it) == 3
        @test collect(it) == [(0, 1), (1, 2), (2, 0)]
    end

    @testset "letter" begin
        using Tenet: letter
        @test all(isletter ∘ only ∘ String, Iterators.map(letter, 1:136104))
    end
end

@testset "Helpers" begin
    @testset "RingPeek" begin
        using Tenet: ringpeek

        it = ringpeek([0 1 2])
        @test length(it) == 3
        @test collect(it) == [(0, 1), (1, 2), (2, 0)]
    end

    @testset "letter" begin
        using Tenet: letter
        # NOTE probabilitic testing due to time taken by `letter`. refactor when `letter` is optimized.
        @test all(isletter ∘ only ∘ String, Iterators.map(letter, rand(1:(Tenet.NUM_UNICODE_LETTERS), 1000)))
    end
end

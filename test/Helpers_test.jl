@testset "Helpers" begin
    @testset "RingPeek" begin
        using Tenet: ringpeek

        it = ringpeek([0 1 2])
        @test length(it) == 3
        @test collect(it) == [(0, 1), (1, 2), (2, 0)]
    end
end
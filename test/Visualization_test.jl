@testset "Visualization" begin
    using CairoMakie
    @testset "plot" begin
        @test plot([0], [0]) isa Any
    end
end
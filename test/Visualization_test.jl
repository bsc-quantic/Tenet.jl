@testset "Visualization" begin
    using GLMakie
    @testset "plot" begin
        @test plot([0], [0]) isa Any
    end
end
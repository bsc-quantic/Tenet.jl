@testset "Lane" begin
    using Tenet: id

    lane = Lane(1)
    @test id(lane) == 1
    @test CartesianIndex(lane) == CartesianIndex(1)

    lane = Lane(1, 2)
    @test id(lane) == (1, 2)
    @test CartesianIndex(lane) == CartesianIndex((1, 2))

    lane = lane"1"
    @test id(lane) == 1
    @test CartesianIndex(lane) == CartesianIndex(1)

    lane = lane"1,2"
    @test id(lane) == (1, 2)
    @test CartesianIndex(lane) == CartesianIndex((1, 2))
end

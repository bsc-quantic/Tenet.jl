@testset "Moment" begin
    using Tenet: id

    moment = Moment(Lane(1), 7)
    @test Lane(moment) == Lane(1)
    @test id(lane) == 1
    @test moment.t == 7

    moment = Moment(Lane(1, 2), 7)
    @test Lane(moment) == Lane(1, 2)
    @test id(lane) == (1, 2)
    @test moment.t == 7

    moment = Moment(lane"1", 7)
    @test Lane(moment) == Lane(1)
    @test id(lane) == 1
    @test moment.t == 7

    moment = Moment(lane"1,2", 7)
    @test Lane(moment) == Lane(1, 2)
    @test id(lane) == (1, 2)
    @test moment.t == 7
end
@testset "Site" begin
    using Tenet: id

    s = Site(1)
    @test Lane(s) == Lane(1)
    @test id(s) == 1
    @test CartesianIndex(s) == CartesianIndex(1)
    @test isdual(s) == false

    s = Site(1; dual=true)
    @test Lane(s) == Lane(1)
    @test id(s) == 1
    @test CartesianIndex(s) == CartesianIndex(1)
    @test isdual(s) == true

    s = Site(1, 2)
    @test Lane(s) == Lane(1, 2)
    @test id(s) == (1, 2)
    @test CartesianIndex(s) == CartesianIndex((1, 2))
    @test isdual(s) == false

    s = Site(1, 2; dual=true)
    @test Lane(s) == Lane(1, 2)
    @test id(s) == (1, 2)
    @test CartesianIndex(s) == CartesianIndex((1, 2))
    @test isdual(s) == true

    s = site"1"
    @test Lane(s) == Lane(1)
    @test id(s) == 1
    @test CartesianIndex(s) == CartesianIndex(1)
    @test isdual(s) == false

    s = site"1'"
    @test Lane(s) == Lane(1)
    @test id(s) == 1
    @test CartesianIndex(s) == CartesianIndex(1)
    @test isdual(s) == true

    s = site"1,2"
    @test Lane(s) == Lane(1, 2)
    @test id(s) == (1, 2)
    @test CartesianIndex(s) == CartesianIndex((1, 2))
    @test isdual(s) == false

    s = site"1,2'"
    @test Lane(s) == Lane(1, 2)
    @test id(s) == (1, 2)
    @test CartesianIndex(s) == CartesianIndex((1, 2))
    @test isdual(s) == true

    s = adjoint(site"1")
    @test Lane(s) == Lane(1)
    @test id(s) == 1
    @test CartesianIndex(s) == CartesianIndex(1)
    @test isdual(s) == true

    s = adjoint(site"1'")
    @test Lane(s) == Lane(1)
    @test id(s) == 1
    @test CartesianIndex(s) == CartesianIndex(1)
    @test isdual(s) == false

    s = adjoint(site"1,2")
    @test Lane(s) == Lane(1, 2)
    @test id(s) == (1, 2)
    @test CartesianIndex(s) == CartesianIndex((1, 2))
    @test isdual(s) == true

    s = adjoint(site"1,2'")
    @test Lane(s) == Lane(1, 2)
    @test id(s) == (1, 2)
    @test CartesianIndex(s) == CartesianIndex((1, 2))
    @test isdual(s) == false
end

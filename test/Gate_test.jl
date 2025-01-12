@testset "Gate" begin
    @testset let data = zeros(2, 2), tensor = Tensor(data, (:i, :j))
        gate = Gate(tensor, [site"1'", site"1"])
        @test gate isa Gate
        @test Tensor(gate) == tensor
        @test Tensor(gate) == tensor
        @test inds(gate) == inds(tensor)
        @test sites(gate) == (site"1'", site"1")
        @test lanes(gate) == (Lane(1),)
        @test sites(gate; set=:inputs) == (site"1'",)
        @test sites(gate; set=:outputs) == (site"1",)
        @test replace(gate, :i => :k) == Gate(Tensor(data, (:k, :j)), [site"1'", site"1"])
        @test replace(gate, site"1'" => :k) == Gate(Tensor(data, (:k, :j)), [site"1'", site"1"])

        gate = Tenet.resetindex(gate)
        @test gate isa Gate
        @test parent(Tensor(gate)) == data
        @test isdisjoint(inds(gate), inds(tensor))
        @test sites(gate) == (site"1'", site"1")

        gate = Gate(data, [site"1'", site"1"])
        @test gate isa Gate
        @test parent(Tensor(gate)) == data
        @test sites(gate) == (site"1'", site"1")
    end
end
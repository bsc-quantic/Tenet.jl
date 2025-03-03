using Test
using Tenet

@testset "Constructors" begin
    @testset "from tensor" begin
        data = zeros(2, 2)
        tensor = Tensor(data, (:i, :j))
        gate = Gate(tensor, [site"1'", site"1"])

        @test gate isa Gate
        @test Tensor(gate) === tensor
    end

    @testset "from array" begin
        data = zeros(2, 2)
        gate = Gate(data, [site"1", site"1'"])

        @test gate isa Gate
        @test parent(Tensor(gate)) == data
        @test issetequal(sites(gate), (site"1", site"1'"))
    end
end

@testset "Tensor Network interface" begin
    data = zeros(2, 2)
    tensor = Tensor(data, (:i, :j))
    gate = Gate(tensor, [site"1", site"1'"])

    @test inds(gate) == inds(tensor)
    @test tensors(gate) == [tensor]
    @test inds(gate; set=:all) == inds(gate; set=:open) == inds(tensor)
    @test isempty(inds(gate; set=:inner)) && isempty(inds(gate; set=:hyper))

    @test replace(gate, :i => :k) == Gate(Tensor(data, (:k, :j)), [site"1", site"1'"])
    @test replace(gate, site"1" => :k) == Gate(Tensor(data, (:k, :j)), [site"1", site"1'"])
end

@testset "Pluggable interface" begin
    @testset let gate = Gate(Tensor(zeros(2, 2), [:i, :j]), [site"1", site"1'"])
        @test sites(gate) == (site"1", site"1'")
        @test sites(gate; set=:inputs) == (site"1'",)
        @test sites(gate; set=:outputs) == (site"1",)

        @test nsites(gate) == 2
        @test nsites(gate; set=:inputs) == 1
        @test nsites(gate; set=:outputs) == 1

        @test inds(gate; at=site"1") == :i
        @test inds(gate; at=site"1'") == :j
        @test issetequal(inds(gate; set=:inputs), [:j])
        @test issetequal(inds(gate; set=:outputs), [:i])

        @test sites(gate; at=:i) == site"1"
        @test sites(gate; at=:j) == site"1'"
    end

    @testset let gate = Gate(Tensor(zeros(2, 2, 2, 2), [:i, :j, :k, :l]), [site"1", site"2", site"1'", site"2'"])
        @test sites(gate) == (site"1", site"2", site"1'", site"2'")
        @test sites(gate; set=:inputs) == (site"1'", site"2'")
        @test sites(gate; set=:outputs) == (site"1", site"2")

        @test nsites(gate) == 4
        @test nsites(gate; set=:inputs) == 2
        @test nsites(gate; set=:outputs) == 2

        @test inds(gate; at=site"1") == :i
        @test inds(gate; at=site"1'") == :k
        @test inds(gate; at=site"2") == :j
        @test inds(gate; at=site"2'") == :l
        @test issetequal(inds(gate; set=:inputs), [:k, :l])
        @test issetequal(inds(gate; set=:outputs), [:i, :j])

        @test sites(gate; at=:i) == site"1"
        @test sites(gate; at=:j) == site"2"
        @test sites(gate; at=:k) == site"1'"
        @test sites(gate; at=:l) == site"2'"
    end
end

@testset "Ansatz interface" begin
    gate = Gate(zeros(2, 2), [site"1", site"1'"])

    @test lanes(gate) == [lane"1"]
    @test nlanes(gate) == 1
end

# resetinds
@testset "resetinds" begin
    data = zeros(2, 2)
    tensor = Tensor(data, (:i, :j))
    gate = Gate(tensor, [site"1", site"1'"])

    gate = Tenet.resetinds(gate)
    @test gate isa Gate
    @test parent(Tensor(gate)) === data
    @test isdisjoint(inds(gate), inds(tensor))
    @test sites(gate) == (site"1", site"1'")
end

using Tenet
using Tenet: Lattice
using BijectiveDicts: BijectiveIdDict
using Graphs

@testset "Ansatz" begin
    # No connectivity
    @testset let tn = TensorNetwork([Tensor(zeros(2), [:i]), Tensor(zeros(2), [:j])]),
        qtn = Quantum(tn, Dict(site"1" => :i, site"2" => :j)),
        graph = Graph(2),
        mapping = BijectiveIdDict{Site,Int}(Pair{Site,Int}[Site(i) => i for i in 1:2]),
        lattice = Lattice(mapping, graph),
        ansatz = Ansatz(qtn, lattice)

        @test zero(ansatz) == Ansatz(zero(qtn), lattice)
        @test Tenet.lattice(ansatz) == lattice
        @test isempty(neighbors(ansatz, site"1"))
        @test !Tenet.has_edge(ansatz, site"1", site"2")

        # the following methods will throw a AssertionError in here, but it's not a hard API requirement
        @test_throws Exception inds(ansatz; bond=(site"1", site"2"))
        @test_throws Exception tensors(ansatz; bond=(site"1", site"2"))
        @test_throws Exception truncate!(ansatz, bond=(site"1", site"2"))

        # TODO test `expect`, `overlap`, `evolve!`, `simple_update!`
    end

    # Complete connectivity
    @testset let n = 2,
        graph = Graphs.complete_graph(2),
        mapping = BijectiveIdDict{Site,Int}(Pair{Site,Int}[Site(i) => i for i in 1:n]),
        lattice = Lattice(mapping, graph),
        tn = TensorNetwork([Tensor(ones(2, 2), [:s1, :i]), Tensor(ones(2, 2), [:s2, :i])]),
        qtn = Quantum(tn, Dict(site"1" => :s1, site"2" => :s2)),
        ansatz = Ansatz(qtn, lattice)

        @test zero(ansatz) == Ansatz(zero(qtn), lattice)
        @test Tenet.lattice(ansatz) == lattice

        @test issetequal(neighbors(ansatz, site"1"), [site"2"])
        @test issetequal(neighbors(ansatz, site"2"), [site"1"])

        @test Tenet.has_edge(ansatz, site"1", site"2")
        @test Tenet.has_edge(ansatz, site"2", site"1")

        @test inds(ansatz; bond=(site"1", site"2")) == :i

        # the following methods will throw a AssertionError in here, but it's not a hard API requirement
        @test_throws Exception tensors(ansatz; bond=(site"1", site"2"))
        @test_throws Exception truncate!(ansatz, bond=(site"1", site"2"))

        # TODO test `expect`, `overlap`, `evolve!`, `simple_update!`
    end
end

using Tenet
using Tenet: Lattice, simple_update!
using BijectiveDicts: BijectiveIdDict
using Graphs
using LinearAlgebra

@testset "Ansatz" begin
    # No connectivity
    @testset let tn = TensorNetwork([Tensor(ones(2), [:i]), Tensor(ones(2), [:j])]),
        qtn = Quantum(tn, Dict(site"1" => :i, site"2" => :j)),
        graph = Graph(2),
        mapping = BijectiveIdDict{Site,Int}(Pair{Site,Int}[Site(i) => i for i in 1:2]),
        lattice = Lattice(mapping, graph),
        ansatz = Ansatz(qtn, lattice)

        @test zero(ansatz) == Ansatz(zero(qtn), lattice)
        @test Tenet.lattice(ansatz) == lattice
        @test isempty(neighbors(ansatz, site"1"))
        @test !Tenet.has_edge(ansatz, site"1", site"2")

        # some AbstractQuantum methods
        @test inds(ansatz; at=site"1") == :i
        @test inds(ansatz; at=site"2") == :j
        @test tensors(ansatz; at=site"1") == Tensor(ones(2), [:i])
        @test tensors(ansatz; at=site"2") == Tensor(ones(2), [:j])

        # the following methods will throw a AssertionError in here, but it's not a hard API requirement
        @test_throws Exception inds(ansatz; bond=(site"1", site"2"))
        @test_throws Exception tensors(ansatz; bond=(site"1", site"2"))
        @test_throws Exception truncate!(ansatz, bond=(site"1", site"2"))

        @test overlap(ansatz, ansatz) ≈ 4.0
        @test norm(ansatz) ≈ 2.0

        @testset "1-local gate" begin
            gate = Quantum(TensorNetwork([Tensor([1 0; 0 0], [:a, :b])]), Dict(site"1'" => :a, site"1" => :b))
            @test tensors(simple_update!(copy(ansatz), gate); at=site"1") ≈ Tensor([1 0], [:i])
            @test tensors(evolve!(copy(ansatz), gate); at=site"1") ≈ Tensor([1 0], [:i])
            @test expect(ansatz, gate) ≈ 2.0
        end

        @testset "2-local gate" begin
            gate = Quantum(
                TensorNetwork([Tensor([1 0; 0 0;;; 0 0; 0 0;;;; 0 0; 0 0;;; 0 0; 0 0], [:a1, :a2, :b1, :b2])]),
                Dict(site"1'" => :a1, site"2'" => :a2, site"1" => :b1, site"2" => :b2),
            )
            @test_throws Exception simple_update!(copy(ansatz), gate)
            @test_throws Exception evolve!(copy(ansatz), gate)
            @test expect(ansatz, gate) ≈ 1.0
        end
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

        @test overlap(ansatz, ansatz) ≈ 16.0
        @test norm(ansatz) ≈ 4.0

        @testset "1-local gate" begin
            gate = Quantum(TensorNetwork([Tensor([1 0; 0 0], [:a, :b])]), Dict(site"1'" => :a, site"1" => :b))
            @test tensors(simple_update!(copy(ansatz), gate); at=site"1") ≈ Tensor([1 0], [:i])
            @test tensors(evolve!(copy(ansatz), gate); at=site"1") ≈ Tensor([1 0], [:i])
            @test expect(ansatz, gate) ≈ 2.0
        end

        @testset "2-local gate" begin
            gate = Quantum(
                TensorNetwork([Tensor([1 0; 0 0;;; 0 0; 0 0;;;; 0 0; 0 0;;; 0 0; 0 0], [:a1, :a2, :b1, :b2])]),
                Dict(site"1'" => :a1, site"2'" => :a2, site"1" => :b1, site"2" => :b2),
            )
            @test_broken simple_update!(copy(ansatz), gate)
            @test_broken evolve!(copy(ansatz), gate)
            @test expect(ansatz, gate) ≈ 4.0
        end
    end
end

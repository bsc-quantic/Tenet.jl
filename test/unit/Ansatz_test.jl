using Tenet
using Tenet: Lattice, simple_update!, AbstractLane
using Graphs
using LinearAlgebra

@testset "Ansatz" begin
    @testset "No connectivity" begin
        n = 2
        tn = TensorNetwork([Tensor(ones(2), [:i]), Tensor(ones(2), [:j])])
        qtn = Quantum(tn, Dict(site"1" => :i, site"2" => :j))

        @test Lattice(Lane.(1:n), Graph(n)) == Lattice(Lane.(1:n))

        lattice = Lattice(Lane.(1:n))
        ansatz = Ansatz(qtn, lattice)

        @test zero(ansatz) == Ansatz(zero(qtn), lattice)
        @test Tenet.lattice(ansatz) == lattice
        @test isempty(neighbors(ansatz, lane"1"))
        @test !has_edge(ansatz, lane"1", lane"2")

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

        @testset let gate = Gate([1 0; 0 0], [site"1'", site"1"])
            @test tensors(simple_update!(copy(ansatz), gate); at=site"1") ≈ Tensor([1, 0], [:i])
            @test tensors(evolve!(copy(ansatz), gate); at=site"1") ≈ Tensor([1, 0], [:i])
            @test expect(ansatz, Quantum(gate)) ≈ 2.0
        end

        @testset let gate = Gate(
                [1 0; 0 0;;; 0 0; 0 0;;;; 0 0; 0 0;;; 0 0; 0 0], [site"1'", site"2'", site"1", site"2"]
            )
            @test_throws Exception simple_update!(copy(ansatz), gate)
            @test_throws Exception evolve!(copy(ansatz), gate)
            @test expect(ansatz, Quantum(gate)) ≈ 1.0
        end
    end

    @testset "Complete connectivity" begin
        n = 2
        graph = Graphs.complete_graph(n)
        lattice = Lattice(Lane.(1:n), graph)
        tn = TensorNetwork([Tensor(ones(2, 2), [:s1, :i]), Tensor(ones(2, 2), [:s2, :i])])
        qtn = Quantum(tn, Dict(site"1" => :s1, site"2" => :s2))
        ansatz = Ansatz(qtn, lattice)

        @test zero(ansatz) == Ansatz(zero(qtn), lattice)
        @test Tenet.lattice(ansatz) == lattice

        @test issetequal(neighbors(ansatz, lane"1"), [lane"2"])
        @test issetequal(neighbors(ansatz, lane"2"), [lane"1"])

        @test has_edge(ansatz, lane"1", lane"2")
        @test has_edge(ansatz, lane"2", lane"1")

        @test inds(ansatz; bond=(lane"1", lane"2")) == :i

        # the following methods will throw a AssertionError in here, but it's not a hard API requirement
        @test_throws Exception tensors(ansatz; bond=(lane"1", lane"2"))
        @test_throws Exception truncate!(ansatz, bond=(lane"1", lane"2"))

        @test overlap(ansatz, ansatz) ≈ 16.0
        @test norm(ansatz) ≈ 4.0

        @testset let gate = Gate([1 0; 0 0], [site"1'", site"1"])
            @test tensors(simple_update!(copy(ansatz), gate); at=site"1") ≈ Tensor([1 1; 0 0], [:s1, :i])
            @test tensors(evolve!(copy(ansatz), gate); at=site"1") ≈ Tensor([1 1; 0 0], [:s1, :i])
            @test expect(ansatz, Quantum(gate)) ≈ 8.0
        end

        @testset let gate = Gate(
                [1 0; 0 0;;; 0 0; 0 0;;;; 0 0; 0 0;;; 0 0; 0 0], [site"1'", site"2'", site"1", site"2"]
            )
            @test expect(ansatz, Quantum(gate)) ≈ 4.0
            @testset let ψ = simple_update!(copy(ansatz), gate)
                @test tensors(ψ; bond=(lane"1", lane"2")) ≈ Tensor([2, 0], [:i])
            end
        end
    end
end

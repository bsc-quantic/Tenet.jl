@testset "Quac" begin
    using Tenet: TensorNetwork, ansatz, Quantum, sites
    using Quac
    n = 2
    qft = Quac.Algorithms.QFT(n)

    @testset "Constructor" begin
        tn = TensorNetwork(qft)

        @test ansatz(tn) == Quantum
        @test tn isa TensorNetwork{Quantum}

        @test issetequal(sites(tn), 1:n)
        @test issetequal(sites(tn, :in), 1:n)
        @test issetequal(sites(tn, :out), 1:n)

        # TODO `physicalinds`,`virtualinds`
        # @test all(isphysical, inds(tn))
    end

    @testset "hcat" begin
        n = 2
        qft = Quac.Algorithms.QFT(n)
        tn = TensorNetwork(qft)

        newtn = hcat(tn, tn)

        @test ansatz(newtn) <: Composite(Quantum, Quantum)
        # @test all(isphysical, inds(newtn))
        @test issetequal(sites(newtn), 1:2)
        @test issetequal(sites(newtn, :in), sites(tn, :in))
        @test issetequal(sites(newtn, :out), sites(tn, :out))
        @test issetequal(labels(newtn, :in), labels(tn, :in))
        @test issetequal(labels(newtn, :out), labels(tn, :out))

        # TODO @test_throws ErrorException ...
    end
end

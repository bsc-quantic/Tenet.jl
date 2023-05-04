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
end

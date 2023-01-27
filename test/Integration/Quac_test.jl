@testset "Quac" begin
    using Tenet: TensorNetwork, ansatz, Quantum, sites, insites, outsites, isphysical
    using Quac
    n = 2
    qft = Quac.Algorithms.QFT(n)

    @testset "Constructor" begin
        tn = TensorNetwork(qft)

        @test ansatz(tn) == Quantum
        @test tn isa TensorNetwork{Quantum}

        # TODO `insites`,`outsites` on `LinearAlgebra.Adjoint(tn)`
        @test issetequal(sites(tn), 1:n)
        @test issetequal(insites(tn), 1:n)
        @test issetequal(outsites(tn), 1:n)

        # TODO `physicalinds`,`virtualinds`
        @test all(isphysical, inds(tn))
    end
end
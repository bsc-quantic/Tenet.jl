@testset "Quantum" begin
    using Tenet: TensorNetwork, ansatz, Quantum, isphysical, sites, insites, insiteinds, outsites, outsiteinds
    using Quac

    @testset "hcat" begin
        n = 2
        qft = Quac.Algorithms.QFT(n)
        tn = TensorNetwork(qft)

        newtn = hcat(tn, tn)

        @test ansatz(newtn) <: Tuple{Quantum,Quantum}
        @test all(isphysical, inds(newtn))
        @test issetequal(sites(newtn), 1:2)
        @test issetequal(insites(newtn), insites(tn))
        @test issetequal(outsites(newtn), outsites(tn))
        @test issetequal(nameof.(insiteinds(newtn)), nameof.(insiteinds(tn)))
        @test issetequal(nameof.(outsiteinds(newtn)), nameof.(outsiteinds(tn)))

        # TODO @test_throws ErrorException ...
    end
end

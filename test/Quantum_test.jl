@testset "Quantum" begin
    using Tenet: TensorNetwork, ansatz, Quantum, sites
    using Quac

    @testset "hcat" begin
        n = 2
        qft = Quac.Algorithms.QFT(n)
        tn = TensorNetwork(qft)

        newtn = hcat(tn, tn)

        @test ansatz(newtn) <: Tuple{Quantum,Quantum}
        # @test all(isphysical, inds(newtn))
        @test issetequal(sites(newtn), 1:2)
        @test issetequal(sites(newtn, :in), sites(tn, :in))
        @test issetequal(sites(newtn, :out), sites(tn, :out))
        @test issetequal(nameof.(labels(newtn, :in)), nameof.(labels(tn, :in)))
        @test issetequal(nameof.(labels(newtn, :out)), nameof.(labels(tn, :out)))

        # TODO @test_throws ErrorException ...
    end
end

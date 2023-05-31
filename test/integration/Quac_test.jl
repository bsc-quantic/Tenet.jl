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
    end

    # TODO currently broken
    # @testset "hcat" begin
    #     n = 2
    #     qft = Quac.Algorithms.QFT(n)
    #     tn = TensorNetwork(qft)

    #     newtn = hcat(tn, tn)

    #     @test ansatz(newtn) <: Composite(Quantum, Quantum)
    #     @test issetequal(sites(newtn), 1:2)

    #     # TODO @test_throws ErrorException ...
    # end
end

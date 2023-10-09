@testset "Quac" begin
    using Quac
    using UUIDs: uuid4

    @testset "Constructor" begin
        n = 2
        qft = Quac.Algorithms.QFT(n)
        tn = QuantumTensorNetwork(qft)

        @test tn isa QuantumTensorNetwork
        @test issetequal(sites(tn), 1:n)
    end

    # TODO currently broken
    @testset "merge" begin
        n = 2
        qft = QuantumTensorNetwork(Quac.Algorithms.QFT(n))
        iqft = replace(qft, [index => Symbol(uuid4()) for index in inds(qft)]...)

        tn = merge(qft, iqft)

        @test tn isa QuantumTensorNetwork
        @test issetequal(sites(tn), 1:2)
    end
end

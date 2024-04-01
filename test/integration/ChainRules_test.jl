@testset "ChainRules" begin
    using Tenet: Tensor, contract
    using ChainRulesTestUtils

    @testset "Tensor" begin
        test_frule(Tensor, fill(1.0), Symbol[])
        test_rrule(Tensor, fill(1.0), Symbol[])

        test_frule(Tensor, fill(1.0, 2), Symbol[:i])
        test_rrule(Tensor, fill(1.0, 2), Symbol[:i])

        test_frule(Tensor, fill(1.0, 2, 3), Symbol[:i, :j])
        test_rrule(Tensor, fill(1.0, 2, 3), Symbol[:i, :j])
    end

    @testset "TensorNetwork" begin
        # TODO it crashes
        # test_frule(TensorNetwork, [])
        # test_rrule(TensorNetwork, [])

        a = Tensor(rand(4, 2), (:i, :j))
        b = Tensor(rand(2, 3), (:j, :k))

        test_frule(TensorNetwork, Tensor[a, b])
        test_rrule(TensorNetwork, Tensor[a, b])
    end

    @testset "conj" begin
        a = Tensor(rand(4, 2), (:i, :j))
        b = Tensor(rand(2, 3), (:j, :k))

        tn = TensorNetwork([a, b])

        @testset "Tensor" begin
            test_frule(Base.conj, a)
            test_rrule(Base.conj, a)
        end

        @testset "TensorNetwork" begin
            test_frule(Base.conj, tn)
            test_rrule(Base.conj, tn)
        end
    end
end

@testset "ChainRules" begin
    using Tenet: Tensor, contract
    using ChainRulesTestUtils

    @testset "Tensor" begin
        test_frule(Tensor, ones(), Symbol[])
        test_rrule(Tensor, ones(), Symbol[])

        test_frule(Tensor, ones(2), Symbol[:i])
        test_rrule(Tensor, ones(2), Symbol[:i])

        test_frule(Tensor, ones(2, 3), Symbol[:i, :j])
        test_rrule(Tensor, ones(2, 3), Symbol[:i, :j])
    end

    @testset "TensorNetwork" begin
        # TODO it crashes
        # test_frule(TensorNetwork, Tensor[])
        # test_rrule(TensorNetwork, Tensor[])

        @testset "equal ndims" begin
            a = Tensor(ones(4, 2), (:i, :j))
            b = Tensor(ones(2, 3), (:j, :k))

            test_frule(TensorNetwork, Tensor[a, b])
            test_rrule(TensorNetwork, Tensor[a, b])
        end

        @testset "different ndims" begin
            a = Tensor(ones(4, 2), (:i, :j))
            b = Tensor(ones(2, 3, 5), (:s, :k, :l))

            test_frule(TensorNetwork, Tensor[a, b])
            test_rrule(TensorNetwork, Tensor[a, b])

            a = Tensor(ones(4, 2), (:i, :j))
            b = Tensor(ones(2, 3, 5), (:j, :k, :l))

            test_frule(TensorNetwork, Tensor[a, b])
            test_rrule(TensorNetwork, Tensor[a, b])
        end
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

    @testset "merge" begin
        a = Tensor(rand(4, 2), (:i, :j))
        b = Tensor(rand(2, 3), (:j, :k))

        test_frule(merge, TensorNetwork([a]), TensorNetwork([b]))
        test_rrule(merge, TensorNetwork([a]), TensorNetwork([b]))
    end
end

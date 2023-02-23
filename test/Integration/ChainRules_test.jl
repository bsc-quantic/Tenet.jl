@testset "ChainRules" begin
    using Tenet: Tensor, contract
    using ChainRulesTestUtils
    ChainRulesTestUtils.ChainRulesCore.debug_mode() = true

    @testset "Tensor" begin
        test_rrule(Tensor, rand(2, 2), (:i, :j))
        test_rrule(Tensor, rand(2, 2), (:i, :j); fkwargs = (; tags = Set(["TEST"])))
    end

    @testset "contract" begin
        @testset "[number product]" begin
            test_rrule(contract, 5.0, 2.0)
            test_frule(contract, 5.0, 2.0)

            # test_rrule(contract, 5, 2)
            # test_frule(contract, 5, 2)
        end

        @testset "[number-tensor product]" begin
            b = Tensor(rand(2, 2), (:i, :j))

            test_frule(contract, 5.0, b)
            test_rrule(contract, 5.0, b)
            # test_rrule(contract, 5.0 + 1im, b)
            # test_rrule(contract, 5, b)
        end

        @testset "ij,ij->" begin
            a = Tensor(rand(2, 2), (:i, :j))
            b = Tensor(rand(2, 2), (:i, :j))

            # test_frule(contract, a, b)
            # test_rrule(contract, a, b) # TODO fix error with FiniteDifferences
        end

        @testset "ik,kj->ij" begin
            a = Tensor(rand(2, 2), (:i, :k))
            b = Tensor(rand(2, 2), (:k, :j))

            test_frule(contract, a, b)
            test_rrule(contract, a, b)
        end
    end
end
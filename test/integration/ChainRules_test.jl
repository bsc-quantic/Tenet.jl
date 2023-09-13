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
end

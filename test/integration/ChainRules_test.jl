@testset "ChainRules" begin
    using Tenet: Tensor, contract
    using Random

    using ChainRulesTestUtils

    using ChainRulesCore
    function ChainRulesTestUtils.rand_tangent(rng::AbstractRNG, x::TensorNetwork)
        return ProjectTo(x)(
            TensorNetwork([ProjectTo(tensor)(rand_tangent.(Ref(rng), tensor)) for tensor in tensors(x)]),
        )
    end

    @testset "Tensor" begin
        test_frule(Tensor, fill(1.0), Symbol[])
        test_rrule(Tensor, fill(1.0), Symbol[])

        test_frule(Tensor, fill(1.0, 2), Symbol[:i])
        test_rrule(Tensor, fill(1.0, 2), Symbol[:i])

        test_frule(Tensor, fill(1.0, 2, 3), Symbol[:i, :j])
        test_rrule(Tensor, fill(1.0, 2, 3), Symbol[:i, :j])
    end
end

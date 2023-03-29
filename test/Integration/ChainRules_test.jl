@testset "ChainRules" begin
    using Tenet: Tensor, contract
    using Random

    using ChainRulesTestUtils
    ChainRulesTestUtils.ChainRulesCore.debug_mode() = true

    using ChainRulesCore
    function ChainRulesTestUtils.rand_tangent(rng::AbstractRNG, x::TensorNetwork)
        return ProjectTo(x)(
            TensorNetwork([ProjectTo(tensor)(rand_tangent.(Ref(rng), tensor)) for tensor in tensors(x)]),
        )
    end

    @testset "contract" begin
        @testset "TensorNetwork" begin
            tn = rand(TensorNetwork, 2, 3)

            @test frule((nothing, tn), only ∘ contract, tn) isa Tuple{eltype(tn),eltype(tn)}
            @test rrule(only ∘ contract, tn) isa Tuple{eltype(tn),Function}

            # TODO FiniteDifferences crashes
            # test_frule(only ∘ contract, tn)
            # test_rrule(only ∘ contract, tn)
        end
    end

    @testset "replace" begin
        using UUIDs: uuid4

        tn = rand(TensorNetwork, 10, 3)
        mapping = [label => Symbol(uuid4()) for label in labels(tn)]

        # TODO fails in check_result.jl@161 -> `c_actual = collect(Broadcast.materialize(actual))`
        # test_rrule(replace, tn, mapping...)
    end
end
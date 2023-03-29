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

            @test frule((nothing, tn), contract, tn) isa Tuple{Tensor{eltype(tn),0},Tensor{eltype(tn),0}}
            @test rrule(contract, tn) isa Tuple{Tensor{eltype(tn),0},Function}

            # TODO FiniteDifferences crashes
            # test_frule(contract, tn)
            # test_rrule(contract, tn)
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
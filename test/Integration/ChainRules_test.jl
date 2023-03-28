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

    @testset "Tensor" begin
        test_rrule(Tensor, rand(2, 2), (:i, :j))
        test_rrule(Tensor, rand(2, 2), (:i, :j); fkwargs = (; tags = Set(["TEST"])))
    end

    @testset "contract" begin
        @testset "[number-number product]" begin
            @testset "float" begin
                test_rrule(contract, 5.0, 2.0)
                test_frule(contract, 5.0, 2.0)
            end

            @testset "complex" begin
                test_rrule(contract, 5.0 + 1.0im, 2.0 - 2.0im)
                test_frule(contract, 5.0 + 1.0im, 2.0 - 2.0im)
            end
        #     @testset "int" begin
        #         test_rrule(contract, 5, 2)
        #         test_frule(contract, 5, 2)
        #     end
        end

        @testset "[number-tensor product]" begin
            b = Tensor(rand(2, 2), (:i, :j))
            z = 1.0 + 1im

            test_frule(contract, 5.0, b)
            test_rrule(contract, 5.0, b)

            test_frule(contract, z, b)
            test_frule(contract, b, z)
            test_rrule(contract, z, b)
            test_rrule(contract, b, z)
        end

        @testset "[adjoint]" begin
            a = Tensor(rand(2, 2), (:i, :j))
            b = adjoint(a)

            test_frule(only ∘ contract, a, b)
            test_rrule(only ∘ contract, a, b)
        end

        # NOTE einsum: ij,ij->
        @testset "[inner product]" begin
            a = Tensor(rand(2, 2), (:i, :j))
            b = Tensor(rand(2, 2), (:i, :j))

            test_frule(only ∘ contract, a, b)
            test_rrule(only ∘ contract, a, b)
        end

        @testset "[outer product]" begin
            a = Tensor(rand(2), (:i,))
            b = Tensor(rand(2), (:j,))

            test_frule(contract, a, b)
            test_rrule(contract, a, b)
        end

        # NOTE einsum: ik,kj->ij
        @testset "[matrix multiplication]" begin
            @testset "[real numbers]" begin
                a = Tensor(rand(2, 2), (:i, :k))
                b = Tensor(rand(2, 2), (:k, :j))

                test_frule(contract, a, b)
                test_rrule(contract, a, b)
            end

            @testset "[complex numbers]" begin
                a = Tensor(rand(Complex{Float64}, 2, 2), (:i, :k))
                b = Tensor(rand(Complex{Float64}, 2, 2), (:k, :j))

                test_frule(contract, a, b)
                test_rrule(contract, a, b)
            end
        end

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
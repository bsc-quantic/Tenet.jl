@testset "Quantum" begin
    state = QuantumTensorNetwork(
        TensorNetwork(Tensor[Tensor(rand(2, 2), (:i, :k)), Tensor(rand(3, 2, 4), (:j, :k, :l))]),
        Symbol[], # input
        [:i, :j], # output
    )

    operator = QuantumTensorNetwork(
        TensorNetwork(Tensor[Tensor(rand(2, 4, 2), (:a, :c, :d)), Tensor(rand(3, 4, 3, 5), (:b, :c, :e, :f))]),
        [:a, :b], # input
        [:d, :e], # output
    )

    @testset "adjoint" begin
        @testset "State" begin
            adj = adjoint(state)
            @test adj.input == state.output
            @test adj.output == state.input
            @test all(((a, b),) -> a == conj(b), zip(tensors(state), tensors(adj)))
        end

        @testset "Operator" begin
            adj = adjoint(operator)
            @test adj.input == operator.output
            @test adj.output == operator.input
            @test all(((a, b),) -> a == conj(b), zip(tensors(operator), tensors(adj)))
        end
    end

    @testset "plug" begin
        @test plug(state) == State()
        @test plug(state') == Dual()
        @test plug(operator) == Operator()
    end

    @testset "sites" begin
        @test issetequal(sites(state), [1, 2])
        @test issetequal(sites(operator), [1, 2])
    end

    @testset "inds" begin
        @testset "State" begin
            @test issetequal(inds(state), [:i, :j, :k, :l])
            @test issetequal(inds(state, set = :open), [:i, :j, :l])
            @test issetequal(inds(state, set = :inner), [:k])
            @test isempty(inds(state, set = :hyper))
            @test isempty(inds(state, set = :in))
            @test issetequal(inds(state, set = :out), [:i, :j])
            @test issetequal(inds(state, set = :physical), [:i, :j])
            @test issetequal(inds(state, set = :virtual), [:k, :l])
        end

        @testset "Operator" begin
            @test issetequal(inds(operator), [:a, :b, :c, :d, :e, :f])
            @test issetequal(inds(operator, set = :open), [:a, :b, :d, :e, :f])
            @test issetequal(inds(operator, set = :inner), [:c])
            @test isempty(inds(operator, set = :hyper))
            @test issetequal(inds(operator, set = :in), [:a, :b])
            @test issetequal(inds(operator, set = :out), [:d, :e])
            @test issetequal(inds(operator, set = :physical), [:a, :b, :d, :e])
            @test issetequal(inds(operator, set = :virtual), [:c, :f])
        end
    end

    @testset "merge" begin
        @testset "(State, State)" begin
            tn = merge(state, state')

            @test plug(tn) == Property()

            @test isempty(sites(tn, :in))
            @test isempty(sites(tn, :out))

            @test isempty(inds(tn, set = :in))
            @test isempty(inds(tn, set = :out))
            @test isempty(inds(tn, set = :physical))
            @test issetequal(inds(tn), inds(tn, set = :virtual))
        end

        @testset "(State, Operator)" begin
            tn = merge(state, operator)

            @test plug(tn) == State()

            @test isempty(sites(tn, :in))
            @test issetequal(sites(tn, :out), sites(operator, :out))

            @test isempty(inds(tn, set = :in))
            @test issetequal(inds(tn, set = :out), inds(operator, :out))
            @test issetequal(inds(tn, set = :physical), inds(operator, :out))
            @test issetequal(inds(tn, set = :virtual), inds(state) ∪ inds(operator, :virtual))
        end

        @testset "(Operator, State)" begin
            tn = merge(operator, state')

            @test plug(tn) == Dual()

            @test issetequal(sites(tn, :in), sites(operator, :in))
            @test isempty(sites(tn, :out))

            @test issetequal(inds(tn, set = :in), inds(operator, :in))
            @test isempty(inds(tn, set = :out))
            @test issetequal(inds(tn, set = :physical), inds(operator, :in))
            @test issetequal(
                inds(tn, set = :virtual),
                inds(state, :virtual) ∪ inds(operator, :virtual) ∪ inds(operator, :out),
            )
        end

        @testset "(Operator, Operator)" begin
            tn = merge(operator, operator')

            @test plug(tn) == Operator()

            @test issetequal(sites(tn, :in), sites(operator, :in))
            @test issetequal(sites(tn, :out), sites(operator, :in))

            @test issetequal(inds(tn, set = :in), inds(operator, :in))
            @test issetequal(inds(tn, set = :out), inds(operator, :in))
            @test issetequal(inds(tn, set = :physical), inds(operator, :in))
            @test inds(operator, :virtual) ⊆ inds(tn, set = :virtual)
        end

        @testset "(Operator, Operator)" begin
            tn = merge(operator', operator)

            @test plug(tn) == Operator()

            @test issetequal(sites(tn, :in), sites(operator, :out))
            @test issetequal(sites(tn, :out), sites(operator, :out))

            @test issetequal(inds(tn, set = :in), inds(operator, :out))
            @test issetequal(inds(tn, set = :out), inds(operator, :out))
            @test issetequal(inds(tn, set = :physical), inds(operator, :out))
            @test inds(operator, :virtual) ⊆ inds(tn, set = :virtual)
        end

        @testset "(State, Operator, State)" begin
            tn = merge(state, operator, state')

            @test plug(tn) == Property()

            @test isempty(sites(tn, :in))
            @test isempty(sites(tn, :out))

            @test isempty(inds(tn, set = :in))
            @test isempty(inds(tn, set = :out))
            @test isempty(inds(tn, set = :physical))
        end
    end
end

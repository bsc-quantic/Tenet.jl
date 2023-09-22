@testset "Quantum" begin
    using Bijections: Bijection

    struct MockState <: Quantum end
    Tenet.plug(::Type{MockState}) = State
    Tenet.metadata(::Type{MockState}) = Tenet.metadata(Quantum)

    struct MockOperator <: Quantum end
    Tenet.plug(::Type{MockOperator}) = Operator
    Tenet.metadata(::Type{MockOperator}) = Tenet.metadata(Quantum)

    state = TensorNetwork{MockState}(
        [Tensor(rand(2, 2), (:i, :k)), Tensor(rand(3, 2, 4), (:j, :k, :l))];
        plug = State,
        plug = [Bijection(Dict([1 => :i, 2 => :j]))],
    )

    operator = TensorNetwork{MockOperator}(
        [Tensor(rand(2, 4, 2), (:a, :c, :d)), Tensor(rand(3, 4, 3, 5), (:b, :c, :e, :f))];
        plug = Operator,
        plug = [Bijection(Dict([1 => :a, 2 => :b])), Bijection(Dict([1 => :d, 2 => :e]))],
    )

    @testset "metadata" begin
        @testset "State" begin
            @test Tenet.checkmeta(state)
            @test hasproperty(state, :plug)
            @test only(state.plug) == Bijection(Dict([1 => :i, 2 => :j]))
        end

        @testset "Operator" begin
            @test Tenet.checkmeta(operator)
            @test hasproperty(operator, :plug)
            @test operator.plug == [Bijection(Dict([1 => :a, 2 => :b])), Bijection(Dict([1 => :d, 2 => :e]))]
        end
    end

    @testset "plug" begin
        @test plug(state) === State

        @test plug(operator) === Operator
    end

    # TODO write tests for
    # - boundary
    # - tensors

    @testset "sites" begin
        @test issetequal(sites(state), [1, 2])
        @test issetequal(sites(operator), [1, 2])
    end

    @testset "inds" begin
        @testset "State" begin
            @test issetequal(inds(state), [:i, :j, :k, :l])
            @test issetequal(inds(state, set = :open), [:i, :j, :l])
            @test issetequal(inds(state, set = :plug), [:i, :j])
            @test issetequal(inds(state, set = :inner), [:k])
            @test isempty(inds(state, set = :hyper))
            @test issetequal(inds(state, set = :virtual), [:k, :l])
        end

        # TODO change the indices
        @testset "Operator" begin
            @test issetequal(inds(operator), [:a, :b, :c, :d, :e, :f])
            @test issetequal(inds(operator, set = :open), [:a, :b, :d, :e, :f])
            @test issetequal(inds(operator, set = :plug), [:a, :b, :d, :e])
            @test issetequal(inds(operator, set = :inner), [:c])
            @test isempty(inds(operator, set = :hyper))
            @test_broken issetequal(inds(operator, set = :virtual), [:c])
        end
    end

    @testset "adjoint" begin
        @testset "State" begin
            adj = adjoint(state)

            @test issetequal(sites(state), sites(adj))
            @test all(i -> inds(state, :plug, i) == inds(adj, :plug, i), sites(state))
        end

        @testset "Operator" begin
            adj = adjoint(operator)

            @test issetequal(sites(operator), sites(adj))
            @test_broken all(i -> inds(operator, :plug, i) == inds(adj, :plug, i), sites(operator))
            @test all(i -> first(operator.plug)[i] == last(adj.plug)[i], sites(operator))
            @test all(i -> last(operator.plug)[i] == first(adj.plug)[i], sites(operator))
        end
    end

    @testset "hcat" begin
        @testset "(State, State)" begin
            expectation = hcat(state, state)
            @test issetequal(sites(expectation), sites(state))
            @test issetequal(inds(expectation, set = :plug), inds(state, set = :plug))
            @test isempty(inds(expectation, set = :open))
            @test issetequal(inds(expectation, set = :inner), inds(expectation, set = :all))
        end

        @testset "(State, Operator)" begin
            expectation = hcat(state, operator)
            @test issetequal(sites(expectation), sites(state))
            @test_broken issetequal(inds(expectation, set = :plug), inds(operator, set = :plug))
            @test_broken isempty(inds(expectation, set = :open))
            @test_broken issetequal(inds(expectation, set = :inner), inds(expectation, set = :all))
        end

        @testset "(Operator, State)" begin
            expectation = hcat(operator, state)
            @test issetequal(sites(expectation), sites(state))
            @test_broken issetequal(inds(expectation, set = :plug), inds(state, set = :plug))
            @test_broken isempty(inds(expectation, set = :open))
            @test_broken issetequal(inds(expectation, set = :inner), inds(expectation, set = :all))
        end

        @testset "(Operator, Operator)" begin
            expectation = hcat(operator, operator)
            @test issetequal(sites(expectation), sites(state))
            @test issetequal(inds(expectation, set = :plug), inds(operator, set = :plug))
            @test isempty(inds(expectation, set = :open))
            @test issetequal(inds(expectation, set = :inner), inds(expectation, set = :all))
        end

        # @testset "(State, Operator, State)" begin
        #     expectation = hcat(state, operator, state')
        #     @test_broken issetequal(sites(expectation), sites(state))
        #     @test_broken issetequal(inds(expectation, set = :plug), inds(operator, set = :plug))
        #     @test_broken isempty(inds(expectation, set = :open))
        #     @test_broken issetequal(inds(expectation, set = :inner), inds(expectation, set = :all))
        # end

        # @testset "(Operator, Operator, Operator)" begin
        #     expectation = hcat(operator, operator, operator)
        #     @test_broken issetequal(sites(expectation), sites(state))
        #     @test_broken issetequal(inds(expectation, set = :plug), inds(operator, set = :plug))
        #     @test_broken isempty(inds(expectation, set = :open))
        #     @test_broken issetequal(inds(expectation, set = :inner), inds(expectation, set = :all))
        # end
    end
end

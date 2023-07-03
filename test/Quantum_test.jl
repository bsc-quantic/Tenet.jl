@testset "Quantum" begin
    using Bijections

    struct MockState <: Quantum end
    Tenet.plug(::Type{MockState}) = State
    Tenet.metadata(::Type{MockState}) = Tenet.metadata(Quantum)

    struct MockOperator <: Quantum end
    Tenet.plug(::Type{MockOperator}) = Operator
    Tenet.metadata(::Type{MockOperator}) = Tenet.metadata(Quantum)

    state = TensorNetwork{MockState}(
        [Tensor(rand(2, 2), (:i, :k)), Tensor(rand(3, 2, 4), (:j, :k, :l))];
        plug = State,
        interlayer = [Bijection(Dict([1 => :i, 2 => :j]))],
    )

    operator = TensorNetwork{MockOperator}(
        [Tensor(rand(2, 4, 2), (:a, :c, :d)), Tensor(rand(3, 4, 2, 5), (:b, :c, :e, :f))];
        plug = Operator,
        interlayer = [Bijection(Dict([1 => :a, 2 => :b])), Bijection(Dict([1 => :d, 2 => :e]))],
    )

    @testset "metadata" begin
        @testset "State" begin
            @test Tenet.checkmeta(state)
            @test hasproperty(state, :interlayer)
            @test only(state.interlayer) == Bijection(Dict([1 => :i, 2 => :j]))
        end

        @testset "Operator" begin
            @test Tenet.checkmeta(operator)
            @test hasproperty(operator, :interlayer)
            @test operator.interlayer == [Bijection(Dict([1 => :a, 2 => :b])), Bijection(Dict([1 => :d, 2 => :e]))]
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

    @testset "labels" begin
        @testset "State" begin
            @test issetequal(labels(state), [:i, :j, :k, :l])
            @test issetequal(labels(state, set = :open), [:i, :j, :l])
            @test issetequal(labels(state, set = :plug), [:i, :j])
            @test issetequal(labels(state, set = :inner), [:k])
            @test isempty(labels(state, set = :hyper))
            @test issetequal(labels(state, set = :virtual), [:k, :l])
        end

        # TODO change the indices
        @testset "Operator" begin
            @test issetequal(labels(operator), [:a, :b, :c, :d, :e, :f])
            @test issetequal(labels(operator, set = :open), [:a, :b, :d, :e, :f])
            @test issetequal(labels(operator, set = :plug), [:a, :b, :d, :e])
            @test issetequal(labels(operator, set = :inner), [:c])
            @test isempty(labels(operator, set = :hyper))
            @test_broken issetequal(labels(operator, set = :virtual), [:c])
        end
    end

    @testset "adjoint" begin
        @testset "State" begin
            adj = adjoint(state)

            @test issetequal(sites(state), sites(adj))
            @test all(i -> labels(state, :plug, i) == labels(adj, :plug, i), sites(state))
        end

        @testset "Operator" begin
            adj = adjoint(operator)

            @test issetequal(sites(operator), sites(adj))
            @test_broken all(i -> labels(operator, :plug, i) == labels(adj, :plug, i), sites(operator))
            @test all(i -> first(operator.interlayer)[i] == last(adj.interlayer)[i], sites(operator))
            @test all(i -> last(operator.interlayer)[i] == first(adj.interlayer)[i], sites(operator))
        end
    end

    @testset "hcat" begin
        @testset "(State, State)" begin
            expectation = hcat(state, state)
            @test issetequal(sites(expectation), sites(state))
            @test issetequal(labels(expectation, set = :plug), labels(state, set = :plug))
            @test isempty(labels(expectation, set = :open))
            @test issetequal(labels(expectation, set = :inner), labels(expectation, set = :all))
        end

        @testset "(State, Operator)" begin
            expectation = hcat(state, operator)
            @test issetequal(sites(expectation), sites(state))
            @test_broken issetequal(labels(expectation, set = :plug), labels(operator, set = :plug))
            @test_broken isempty(labels(expectation, set = :open))
            @test_broken issetequal(labels(expectation, set = :inner), labels(expectation, set = :all))
        end

        @testset "(Operator, State)" begin
            expectation = hcat(operator, state)
            @test issetequal(sites(expectation), sites(state))
            @test_broken issetequal(labels(expectation, set = :plug), labels(state, set = :plug))
            @test_broken isempty(labels(expectation, set = :open))
            @test_broken issetequal(labels(expectation, set = :inner), labels(expectation, set = :all))
        end

        @testset "(Operator, Operator)" begin
            expectation = hcat(operator, operator)
            @test issetequal(sites(expectation), sites(state))
            @test issetequal(labels(expectation, set = :plug), labels(operator, set = :plug))
            @test isempty(labels(expectation, set = :open))
            @test issetequal(labels(expectation, set = :inner), labels(expectation, set = :all))
        end

        # @testset "(State, Operator, State)" begin
        #     expectation = hcat(state, operator, state')
        #     @test_broken issetequal(sites(expectation), sites(state))
        #     @test_broken issetequal(labels(expectation, set = :plug), labels(operator, set = :plug))
        #     @test_broken isempty(labels(expectation, set = :open))
        #     @test_broken issetequal(labels(expectation, set = :inner), labels(expectation, set = :all))
        # end

        # @testset "(Operator, Operator, Operator)" begin
        #     expectation = hcat(operator, operator, operator)
        #     @test_broken issetequal(sites(expectation), sites(state))
        #     @test_broken issetequal(labels(expectation, set = :plug), labels(operator, set = :plug))
        #     @test_broken isempty(labels(expectation, set = :open))
        #     @test_broken issetequal(labels(expectation, set = :inner), labels(expectation, set = :all))
        # end
    end
end

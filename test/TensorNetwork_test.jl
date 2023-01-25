@testset "TensorNetwork" begin
    using Tenet: TensorNetwork, ansatz, Arbitrary, tensors, inds, labels, openinds, hyperinds, Index

    # TODO refactor these tests
    let tn = TensorNetwork()
        @test ansatz(tn) == ansatz(typeof(tn)) == Tenet.Arbitrary
        @test isempty(tensors(tn))
        @test isempty(inds(tn))
        @test isempty(size(tn))

        tensor = Tensor(zeros(2, 3), (:i, :j))
        push!(tn, tensor)
        @test length(tn) == 1
        @test issetequal(labels(tn), [:i, :j])
        @test issetequal(inds(tn), [Index(:i, 2), Index(:j, 3)])
        @test size(tn) == Dict(:i => 2, :j => 3)
        @test issetequal(openinds(tn), [Index(:i, 2), Index(:j, 3)])
        @test isempty(hyperinds(tn))

        pop!(tn, tensor)
        @test length(tn) == 0
        @test isempty(tensors(tn))
        @test isempty(inds(tn))
        @test isempty(size(tn))
    end

    @test_throws DimensionMismatch begin
        tn = TensorNetwork()
        tensor = Tensor(zeros(2, 3), (:i, :i))
        push!(tn, tensor)
    end

    let tn = TensorNetwork()
        tensor = Tensor(zeros(2, 2, 2), (:i, :i, :i))
        push!(tn, tensor)

        @test issetequal(labels(tn), [:i])
        @test issetequal(hyperinds(tn), [Index(:i, 2)])

        delete!(tn, :i)
        @test isempty(tensors(tn))
        @test isempty(inds(tn))
    end

    let tn = rand(TensorNetwork, 10, 3)
        @test tn isa TensorNetwork{Arbitrary}
        @test length(tn) == 10
    end

    @testset "copy" begin
        tn = rand(TensorNetwork, 10, 3)
        tn_copy = copy(tn)

        @test tensors(tn_copy) !== tensors(tn)
        @test all(tensors(tn_copy) .=== tensors(tn))
        @test inds(tn_copy) !== inds(tn)
        @test all(inds(tn_copy) .== inds(tn))
        # TODO test index metadata is copied
    end
end
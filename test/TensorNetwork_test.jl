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

    @testset "inds" begin
        using Tenet: openinds, innerinds, hyperinds

        tn = TensorNetwork([
            Tensor(zeros(2, 2), (:i, :j)),
            Tensor(zeros(2, 2), (:i, :k)),
            Tensor(zeros(2, 2, 2), (:i, :l, :m)),
            Tensor(zeros(2, 2), (:l, :m)),
        ])

        @test issetequal(labels(tn), (:i, :j, :k, :l, :m))
        @test issetequal(inds(tn), [Index(i, 2) for i in (:i, :j, :k, :l, :m)])
        @test issetequal(openinds(tn) .|> nameof, (:j, :k))
        @test issetequal(innerinds(tn) .|> nameof, (:i, :l, :m))
        @test issetequal(hyperinds(tn) .|> nameof, (:i,))
    end

    @testset "size" begin
        tn = TensorNetwork([
            Tensor(zeros(2, 3), (:i, :j)),
            Tensor(zeros(2, 4), (:i, :k)),
            Tensor(zeros(2, 5, 6), (:i, :l, :m)),
            Tensor(zeros(5, 6), (:l, :m)),
        ])

        @test size(tn) == Dict((:i => 2, :j => 3, :k => 4, :l => 5, :m => 6))
        @test all([size(tn, :i) == 2, size(tn, :j) == 3, size(tn, :k) == 4, size(tn, :l) == 5, size(tn, :m) == 6])
    end

    @testset "select" begin
        using Tenet: select

        t_ij = Tensor(zeros(2, 2), (:i, :j))
        t_ik = Tensor(zeros(2, 2), (:i, :k))
        t_ilm = Tensor(zeros(2, 2, 2), (:i, :l, :m))
        t_lm = Tensor(zeros(2, 2), (:l, :m))
        tn = TensorNetwork([t_ij, t_ik, t_ilm, t_lm])

        @test issetequal(select(tn, :i), (t_ij, t_ik, t_ilm))
        @test issetequal(select(tn, :j), (t_ij,))
        @test issetequal(select(tn, :k), (t_ik,))
        @test issetequal(select(tn, :l), (t_ilm, t_lm))
        @test issetequal(select(tn, :m), (t_ilm, t_lm))
        @test issetequal(select(tn, (:i, :j)), (t_ij,))
        @test issetequal(select(tn, (:i, :k)), (t_ik,))
        @test issetequal(select(tn, (:i, :l)), (t_ilm,))
        @test issetequal(select(tn, (:l, :m)), (t_ilm, t_lm))
        @test_throws KeyError select(tn, :_)
        @test isempty(select(tn, (:j, :l)))
    end
end
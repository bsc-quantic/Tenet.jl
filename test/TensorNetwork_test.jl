@testset "TensorNetwork" begin
    using Tenet: TensorNetwork, ansatz, Arbitrary, tensors, inds, labels, openinds, hyperinds, Index

    @testset "Constructors" begin
        @testset "empty" begin
            tn = TensorNetwork()
            @test ansatz(tn) == ansatz(typeof(tn)) == Tenet.Arbitrary
            @test isempty(tensors(tn))
            @test isempty(inds(tn))
            @test isempty(size(tn))
        end

        @testset "list" begin
            tensor = Tensor(zeros(2, 3), (:i, :j))
            tn = TensorNetwork([tensor])

            @test only(tensors(tn)) === tensor

            @test length(tn) == 1
            @test issetequal(labels(tn), [:i, :j])
            @test issetequal(inds(tn), [Index(:i, 2), Index(:j, 3)])
            @test size(tn) == Dict(:i => 2, :j => 3)
            @test issetequal(openinds(tn), [Index(:i, 2), Index(:j, 3)])
        end
    end

    @testset "push!" begin
        tn = TensorNetwork()
        tensor = Tensor(zeros(2, 2, 2), (:i, :j, :k))

        push!(tn, tensor)
        @test length(tn) == 1
        @test issetequal(labels(tn), [:i, :j, :k])
        @test issetequal(inds(tn), [Index(:i, 2), Index(:j, 2), Index(:k, 2)])
        @test size(tn) == Dict(:i => 2, :j => 2, :k => 2)
        @test issetequal(openinds(tn), [Index(:i, 2), Index(:j, 2), Index(:k, 2)])
        @test isempty(hyperinds(tn))
    end

    @test_throws DimensionMismatch begin
        tn = TensorNetwork()
        tensor = Tensor(zeros(2, 3), (:i, :i))
        push!(tn, tensor)
    end

    @testset "append!" begin
        tensor = Tensor(zeros(2, 3), (:i, :j))
        A = TensorNetwork()
        B = TensorNetwork()

        append!(B, [tensor])
        @test only(tensors(B)) === tensor

        append!(A, B)
        @test only(tensors(A)) === tensor
    end

    @testset "pop!" begin
        @testset "by reference" begin
            tensor = Tensor(zeros(2, 3), (:i, :j))
            tn = TensorNetwork([tensor])

            @test pop!(tn, tensor) === tensor
            @test length(tn) == 0
            @test isempty(tensors(tn))
            @test isempty(inds(tn))
            @test isempty(size(tn))
        end

        @testset "by symbol" begin
            tensor = Tensor(zeros(2, 3), (:i, :j))
            tn = TensorNetwork([tensor])

            @test only(pop!(tn, :i)) === tensor
            @test length(tn) == 0
            @test isempty(tensors(tn))
            @test isempty(inds(tn))
            @test isempty(size(tn))
        end

        @testset "by symbols" begin
            tensor = Tensor(zeros(2, 3), (:i, :j))
            tn = TensorNetwork([tensor])

            @test only(pop!(tn, (:i, :j))) === tensor
            @test length(tn) == 0
            @test isempty(tensors(tn))
            @test isempty(inds(tn))
            @test isempty(size(tn))
        end
    end

    # TODO by simbols
    @testset "delete!" begin
        tensor = Tensor(zeros(2, 3), (:i, :j))
        tn = TensorNetwork([tensor])

        @test delete!(tn, tensor) === tn
        @test length(tn) == 0
        @test isempty(tensors(tn))
        @test isempty(inds(tn))
        @test isempty(size(tn))
    end

    @testset "hyperinds" begin
        tn = TensorNetwork()
        tensor = Tensor(zeros(2, 2, 2), (:i, :i, :i))
        push!(tn, tensor)

        @test issetequal(labels(tn), [:i])
        @test issetequal(hyperinds(tn), [Index(:i, 2)])

        delete!(tn, :i)
        @test isempty(tensors(tn))
        @test isempty(inds(tn))
    end

    @testset "rand" begin
        tn = rand(TensorNetwork, 10, 3)
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

    @testset "selectdim" begin
        tn = rand(TensorNetwork, 10, 3)
        label = first(labels(tn))

        @test label ∉ labels(selectdim(tn, label, 1))
        @test label ∈ labels(selectdim(tn, label, 1:1))
        @test size(selectdim(tn, label, 1:1), label) == 1
    end

    @testset "view" begin
        tn = rand(TensorNetwork, 10, 3)
        targets = labels(tn)[1:3]

        slice = @view tn[[label => 1 for label in targets]...]
        @test isdisjoint(targets, labels(slice))

        slice = @view tn[[label => 1:1 for label in targets]...]
        @test targets ⊆ labels(slice)
    end

    @testset "Base.replace!" begin
        using Tenet: openinds, hyperinds, select

        t_ij = Tensor(zeros(2, 2), (:i, :j); tags = Set{String}(["TEST"]))
        t_ik = Tensor(zeros(2, 2), (:i, :k))
        t_ilm = Tensor(zeros(2, 2, 2), (:i, :l, :m))
        t_lm = Tensor(zeros(2, 2), (:l, :m))
        tn = TensorNetwork([t_ij, t_ik, t_ilm, t_lm])

        @testset "replace labels" begin
            mapping = (:i => :u, :j => :v, :k => :w, :l => :x, :m => :y)
            replace!(tn, mapping...)

            @test issetequal(labels(tn), (:u, :v, :w, :x, :y))
            @test issetequal(openinds(tn) .|> nameof, (:v, :w))
            @test issetequal(innerinds(tn) .|> nameof, (:u, :x, :y))
            @test issetequal(hyperinds(tn) .|> nameof, (:u,))

            @test only(select(tn, (:u, :v))) == replace(t_ij, mapping...)
            @test only(select(tn, (:u, :w))) == replace(t_ik, mapping...)
            @test only(select(tn, (:u, :x, :y))) == replace(t_ilm, mapping...)

            @test hastag(only(select(tn, (:u, :v))), "TEST")
        end

        @testset "replace tensors" begin
            old_tensor = tensors(tn, 2)

            @test_throws ArgumentError begin
                new_tensor = Tensor(rand(2, 2), (:a, :b))
                replace!(tn, old_tensor => new_tensor)
            end

            new_tensor = Tensor(rand(2, 2), (:u, :w))

            replace!(tn, old_tensor => new_tensor)
            @test new_tensor === tensors(tn, 2)

            # Check if connections are maintained
            for label in labels(new_tensor)
                index = tn.inds[label]
                @test new_tensor in index.links
                @test !(old_tensor in index.links)
            end

            # New tensor network with two tensors with the same labels
            A = Tensor(rand(2, 2), (:u, :w))
            B = Tensor(rand(2, 2), (:u, :w))
            tn = TensorNetwork([A, B])

            new_tensor = Tensor(rand(2, 2), (:u, :w))

            replace!(tn, B => new_tensor)
            @test A === tensors(tn, 1)
            @test new_tensor === tensors(tn, 2)

            tn = TensorNetwork([A, B])
            replace!(tn, A => new_tensor)
            @test new_tensor === tensors(tn, 1)
            @test B === tensors(tn_2, 2)
        end
    end
end
@testset "TensorNetwork" begin
    @testset "Constructors" begin
        @testset "empty" begin
            tn = TensorNetwork()
            @test ansatz(tn) == ansatz(typeof(tn)) === Tenet.Arbitrary
            @test isempty(tensors(tn))
            @test isempty(labels(tn))
            @test isempty(size(tn))
        end

        @testset "list" begin
            tensor = Tensor(zeros(2, 3), (:i, :j))
            tn = TensorNetwork([tensor])

            @test only(tensors(tn)) === tensor

            @test length(tn) == 1
            @test issetequal(labels(tn), [:i, :j])
            @test size(tn) == Dict(:i => 2, :j => 3)
            @test issetequal(labels(tn, :open), [:i, :j])
            @test isempty(labels(tn, :hyper))

            tensor1 = Tensor(zeros(2, 2), (:i, :j))
            tensor2 = Tensor(zeros(3, 3), (:j, :k))
            @test_throws DimensionMismatch tn = TensorNetwork([tensor1, tensor2])
        end
    end

    @testset "push!" begin
        tn = TensorNetwork()
        tensor = Tensor(zeros(2, 2, 2), (:i, :j, :k))

        push!(tn, tensor)
        @test length(tn) == 1
        @test issetequal(labels(tn), [:i, :j, :k])
        @test size(tn) == Dict(:i => 2, :j => 2, :k => 2)
        @test issetequal(labels(tn, :open), [:i, :j, :k])
        @test isempty(labels(tn, :hyper))

        @test_throws DimensionMismatch push!(tn, Tensor(zeros(3, 3), (:i, :j)))
    end

    @test_throws Exception begin
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
            @test isempty(size(tn))
        end

        @testset "by symbol" begin
            tensor = Tensor(zeros(2, 3), (:i, :j))
            tn = TensorNetwork([tensor])

            @test only(pop!(tn, :i)) === tensor
            @test length(tn) == 0
            @test isempty(tensors(tn))
            @test isempty(size(tn))
        end

        @testset "by symbols" begin
            tensor = Tensor(zeros(2, 3), (:i, :j))
            tn = TensorNetwork([tensor])

            @test only(pop!(tn, (:i, :j))) === tensor
            @test length(tn) == 0
            @test isempty(tensors(tn))
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
        @test isempty(size(tn))
    end

    @testset "hyperinds" begin
        tn = TensorNetwork()
        tensor = Tensor(zeros(2, 2, 2), (:i, :i, :i))
        push!(tn, tensor)

        @test issetequal(labels(tn), [:i])
        @test issetequal(labels(tn, :hyper), [:i])

        delete!(tn, :i)
        @test isempty(tensors(tn))
    end

    @testset "rand" begin
        tn = rand(TensorNetwork, 10, 3)
        @test tn isa TensorNetwork{Arbitrary}
        @test length(tn) == 10
    end

    @testset "copy" begin
        tn = rand(TensorNetwork, 10, 3)
        tn_copy = copy(tn)

        @test tensors(tn_copy) !== tensors(tn) && all(tensors(tn_copy) .=== tensors(tn))
        @test labels(tn) !== labels(tn_copy) && issetequal(labels(tn), labels(tn_copy))
    end

    @testset "labels" begin
        tn = TensorNetwork([
            Tensor(zeros(2, 2), (:i, :j)),
            Tensor(zeros(2, 2), (:i, :k)),
            Tensor(zeros(2, 2, 2), (:i, :l, :m)),
            Tensor(zeros(2, 2), (:l, :m)),
        ])

        @test issetequal(labels(tn), (:i, :j, :k, :l, :m))
        @test issetequal(labels(tn, :open), (:j, :k))
        @test issetequal(labels(tn, :inner), (:i, :l, :m))
        @test issetequal(labels(tn, :hyper), (:i,))
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

    # @testset "selectdim" begin
    #     tn = rand(TensorNetwork, 10, 3)
    #     label = first(labels(tn))

    #     @test label ∉ labels(view(tn, label => 1))
    #     @test label ∈ labels(view(tn, label => 1:1))
    #     @test size(view(tn, label => 1:1), label) == 1
    # end

    @testset "view" begin
        tn = rand(TensorNetwork, 10, 3, seed = 1)
        targets = labels(tn)[1:3]

        slice = @view tn[[label => 1 for label in targets]...]
        @test isdisjoint(targets, labels(slice))

        slice = @view tn[[label => 1:1 for label in targets]...]
        @test targets ⊆ labels(slice)
    end

    @testset "contract" begin
        tn = rand(TensorNetwork, 5, 3)
        @test contract(tn) isa Tensor

        A = Tensor(rand(2, 2, 2), (:i, :j, :k))
        B = Tensor(rand(2, 2, 2), (:k, :l, :m))
        tn = TensorNetwork([A, B])
        @test contract(tn) isa Tensor
    end

    @testset "Base.replace!" begin
        t_ij = Tensor(zeros(2, 2), (:i, :j); tags = Set{String}(["TEST"]))
        t_ik = Tensor(zeros(2, 2), (:i, :k))
        t_ilm = Tensor(zeros(2, 2, 2), (:i, :l, :m))
        t_lm = Tensor(zeros(2, 2), (:l, :m))
        tn = TensorNetwork([t_ij, t_ik, t_ilm, t_lm])

        @testset "replace labels" begin
            mapping = (:i => :u, :j => :v, :k => :w, :l => :x, :m => :y)

            @test_throws ArgumentError replace!(tn, :i => :j, :k => :l)
            replace!(tn, mapping...)

            @test issetequal(labels(tn), (:u, :v, :w, :x, :y))
            @test issetequal(labels(tn, :open), (:v, :w))
            @test issetequal(labels(tn, :inner), (:u, :x, :y))
            @test issetequal(labels(tn, :hyper), (:u,))

            @test only(select(tn, (:u, :v))) == replace(t_ij, mapping...)
            @test only(select(tn, (:u, :w))) == replace(t_ik, mapping...)
            @test only(select(tn, (:u, :x, :y))) == replace(t_ilm, mapping...)

            @test hastag(only(select(tn, (:u, :v))), "TEST")
        end

        @testset "replace tensors" begin
            old_tensor = tn.tensors[2]

            @test_throws ArgumentError begin
                new_tensor = Tensor(rand(2, 2), (:a, :b))
                replace!(tn, old_tensor => new_tensor)
            end

            new_tensor = Tensor(rand(2, 2), (:u, :w))

            replace!(tn, old_tensor => new_tensor)
            @test new_tensor === tn.tensors[2]

            # Check if connections are maintained
            # for label in labels(new_tensor)
            #     index = tn.inds[label]
            #     @test new_tensor in index.links
            #     @test !(old_tensor in index.links)
            # end

            # New tensor network with two tensors with the same labels
            A = Tensor(rand(2, 2), (:u, :w))
            B = Tensor(rand(2, 2), (:u, :w))
            tn = TensorNetwork([A, B])

            new_tensor = Tensor(rand(2, 2), (:u, :w))

            replace!(tn, B => new_tensor)
            @test A === tn.tensors[1]
            @test new_tensor === tn.tensors[2]

            tn = TensorNetwork([A, B])
            replace!(tn, A => new_tensor)

            @test issetequal(tensors(tn), [new_tensor, B])

            # Test chain of replacements
            A = Tensor(zeros(2, 2), (:i, :j))
            B = Tensor(zeros(2, 2), (:j, :k))
            C = Tensor(zeros(2, 2), (:k, :l))
            tn = TensorNetwork([A, B, C])

            @test_throws ArgumentError replace!(tn, A => B, B => C, C => A)

            new_tensor = Tensor(rand(2, 2), (:i, :j))
            new_tensor2 = Tensor(ones(2, 2), (:i, :j))

            replace!(tn, A => new_tensor, new_tensor => new_tensor2)
            @test issetequal(tensors(tn), [new_tensor2, B, C])
        end
    end
end

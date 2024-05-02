@testset "TensorNetwork" begin
    @testset "Constructors" begin
        @testset "empty" begin
            tn = TensorNetwork()
            @test isempty(tensors(tn))
            @test isempty(inds(tn))
            @test isempty(size(tn))
        end

        @testset "list" begin
            tensor = Tensor(zeros(2, 3), (:i, :j))
            tn = TensorNetwork([tensor])

            @test only(tensors(tn)) === tensor
            @test issetequal(inds(tn), [:i, :j])
            @test size(tn) == Dict(:i => 2, :j => 3)
            @test issetequal(inds(tn; set=:open), [:i, :j])
            @test isempty(inds(tn; set=:hyper))
        end

        @testset "TensorNetwork with tensors of different dimensions" begin
            tensor1 = Tensor(zeros(2, 2), (:i, :j))
            tensor2 = Tensor(zeros(3, 3), (:j, :k))
            @test_skip @test_throws DimensionMismatch tn = TensorNetwork([tensor1, tensor2])
        end
    end

    @testset "push!" begin
        tn = TensorNetwork()
        tensor = Tensor(zeros(2, 2, 2), (:i, :j, :k))

        push!(tn, tensor)

        @test length(tensors(tn)) == 1
        @test issetequal(inds(tn), [:i, :j, :k])
        @test size(tn) == Dict(:i => 2, :j => 2, :k => 2)
        @test issetequal(inds(tn; set=:open), [:i, :j, :k])
        @test isempty(inds(tn; set=:hyper))

        @test_throws DimensionMismatch push!(tn, Tensor(zeros(3, 3), (:i, :j)))

        @test_throws Exception begin
            tn = TensorNetwork()
            tensor = Tensor(zeros(2, 3), (:i, :i))
            push!(tn, tensor)
        end
    end

    @testset "append!" begin
        tensor = Tensor(zeros(2, 3), (:i, :j))
        A = TensorNetwork()
        B = TensorNetwork()

        append!(B, [tensor])
        @test only(tensors(B)) === tensor
    end

    @testset "merge!" begin
        tensor = Tensor(zeros(2, 3), (:i, :j))
        A = TensorNetwork([tensor])
        B = TensorNetwork()

        merge!(A, B)
        @test only(tensors(A)) === tensor
    end

    @testset "pop!" begin
        @testset "by reference" begin
            tensor = Tensor(zeros(2, 3), (:i, :j))
            tn = TensorNetwork([tensor])

            @test pop!(tn, tensor) === tensor
            @test length(tensors(tn)) == 0
            @test isempty(tensors(tn))
            @test isempty(size(tn))
        end

        @testset "by symbol" begin
            tensor = Tensor(zeros(2, 3), (:i, :j))
            tn = TensorNetwork([tensor])

            @test only(pop!(tn, :i)) === tensor
            @test length(tensors(tn)) == 0
            @test isempty(tensors(tn))
            @test isempty(size(tn))
        end

        @testset "by symbols" begin
            tensor = Tensor(zeros(2, 3), (:i, :j))
            tn = TensorNetwork([tensor])

            @test only(pop!(tn, (:i, :j))) === tensor
            @test length(tensors(tn)) == 0
            @test isempty(tensors(tn))
            @test isempty(size(tn))
        end
    end

    # TODO by simbols
    @testset "delete!" begin
        tensor = Tensor(zeros(2, 3), (:i, :j))
        tn = TensorNetwork([tensor])

        @test delete!(tn, tensor) === tn
        @test length(tensors(tn)) == 0
        @test isempty(tensors(tn))
        @test isempty(size(tn))
    end

    @testset "hyperinds" begin
        @test begin
            tn = TensorNetwork([Tensor(zeros(2), (:i,)), Tensor(zeros(2), (:i,)), Tensor(zeros(2), (:i,))])

            issetequal(inds(tn; set=:hyper), [:i])
        end

        @test begin
            tensor = Tensor(zeros(2, 2, 2), (:i, :i, :i))
            tn = TensorNetwork([tensor])

            issetequal(inds(tn; set=:hyper), [:i])
        end

        @test_broken begin
            tensor = Tensor(zeros(2, 2, 2), (:i, :i, :i))
            tn = TensorNetwork()
            push!(tn, tensor)

            issetequal(inds(tn; set=:hyper), [:i])
        end
    end

    @testset "rand" begin
        tn = rand(TensorNetwork, 10, 3)
        @test tn isa TensorNetwork
        @test length(tensors(tn)) == 10
    end

    @testset "copy" begin
        tensor = Tensor(zeros(2, 2), (:i, :j))
        tn = TensorNetwork([tensor])
        tn_copy = copy(tn)

        @test tensors(tn_copy) !== tensors(tn) && all(tensors(tn_copy) .=== tensors(tn))
        @test inds(tn) !== inds(tn_copy) && issetequal(inds(tn), inds(tn_copy))
    end

    @testset "inds" begin
        tn = TensorNetwork([
            Tensor(zeros(2, 2), (:i, :j)),
            Tensor(zeros(2, 2), (:i, :k)),
            Tensor(zeros(2, 2, 2), (:i, :l, :m)),
            Tensor(zeros(2, 2), (:l, :m)),
        ],)

        @test issetequal(inds(tn), [:i, :j, :k, :l, :m])
        @test issetequal(inds(tn; set=:open), [:j, :k])
        @test issetequal(inds(tn; set=:inner), [:i, :l, :m])
        @test issetequal(inds(tn; set=:hyper), [:i])
    end

    @testset "size" begin
        tn = TensorNetwork([
            Tensor(zeros(2, 3), (:i, :j)),
            Tensor(zeros(2, 4), (:i, :k)),
            Tensor(zeros(2, 5, 6), (:i, :l, :m)),
            Tensor(zeros(5, 6), (:l, :m)),
        ],)

        @test size(tn) == Dict((:i => 2, :j => 3, :k => 4, :l => 5, :m => 6))
        @test all([size(tn, :i) == 2, size(tn, :j) == 3, size(tn, :k) == 4, size(tn, :l) == 5, size(tn, :m) == 6])
    end

    @testset "tensors" begin
        t_ij = Tensor(zeros(2, 2), (:i, :j))
        t_ik = Tensor(zeros(2, 2), (:i, :k))
        t_ilm = Tensor(zeros(2, 2, 2), (:i, :l, :m))
        t_lm = Tensor(zeros(2, 2), (:l, :m))
        tn = TensorNetwork([t_ij, t_ik, t_ilm, t_lm])

        @test issetequal(tensors(tn, :any, :i), (t_ij, t_ik, t_ilm))
        @test issetequal(tensors(tn, :any, :j), (t_ij,))
        @test issetequal(tensors(tn, :any, :k), (t_ik,))
        @test issetequal(tensors(tn, :any, :l), (t_ilm, t_lm))
        @test issetequal(tensors(tn, :any, :m), (t_ilm, t_lm))
        @test issetequal(tensors(tn, :containing, (:i, :j)), (t_ij,))
        @test issetequal(tensors(tn, :containing, (:i, :k)), (t_ik,))
        @test issetequal(tensors(tn, :containing, (:i, :l)), (t_ilm,))
        @test issetequal(tensors(tn, :containing, (:l, :m)), (t_ilm, t_lm))
        @test_throws KeyError tensors(tn, :any, :_)
        @test isempty(tensors(tn, :containing, (:j, :l)))
    end

    @testset "getindex" begin
        t_ij = Tensor(zeros(2, 2), (:i, :j))
        t_ik = Tensor(zeros(2, 2), (:i, :k))
        t_ilm = Tensor(zeros(2, 2, 2), (:i, :l, :m))
        t_lm = Tensor(zeros(2, 2), (:l, :m))
        tn = TensorNetwork([t_ij, t_ik, t_ilm, t_lm])

        @test t_ij === tn[:i, :j]
        @test t_ik === tn[:i, :k]
        @test t_ilm === tn[:i, :l, :m]
        @test t_lm === tn[:l, :m]

        # NOTE although it should throw `KeyError`, it throws `ArgumentError` due to implementation 
        @test_throws ArgumentError tn[:i, :x]
        @test_throws ArgumentError tn[:i, :j, :k]
    end

    # @testset "selectdim" begin
    #     tn = rand(TensorNetwork, 10, 3)
    #     label = first(inds(tn))

    #     @test label ∉ inds(view(tn, label => 1))
    #     @test label ∈ inds(view(tn, label => 1:1))
    #     @test size(view(tn, label => 1:1), label) == 1
    # end

    @testset "view" begin
        tn = rand(TensorNetwork, 10, 3; seed=1)
        targets = inds(tn)[1:3]

        slice = @view tn[[label => 1 for label in targets]...]
        @test isdisjoint(targets, inds(slice))

        slice = @view tn[[label => 1:1 for label in targets]...]
        @test targets ⊆ inds(slice)
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
        @testset "replace inds" begin
            t_ij = Tensor(zeros(2, 2), (:i, :j))
            t_ik = Tensor(zeros(2, 2), (:i, :k))
            t_ilm = Tensor(zeros(2, 2, 2), (:i, :l, :m))
            t_lm = Tensor(zeros(2, 2), (:l, :m))
            tn = TensorNetwork([t_ij, t_ik, t_ilm, t_lm])

            mapping = (:i => :u, :j => :v, :k => :w, :l => :x, :m => :y)

            @test_throws ArgumentError replace!(tn, :i => :j, :k => :l)
            replace!(tn, mapping...)

            @test issetequal(inds(tn), (:u, :v, :w, :x, :y))
            @test issetequal(inds(tn; set=:open), (:v, :w))
            @test issetequal(inds(tn; set=:inner), (:u, :x, :y))
            @test issetequal(inds(tn; set=:hyper), (:u,))

            @test only(tensors(tn, :containing, (:u, :v))) == replace(t_ij, mapping...)
            @test only(tensors(tn, :containing, (:u, :w))) == replace(t_ik, mapping...)
            @test only(tensors(tn, :containing, (:u, :x, :y))) == replace(t_ilm, mapping...)
        end

        @testset "replace tensors" begin
            t_ij = Tensor(zeros(2, 2), (:i, :j))
            t_ik = Tensor(zeros(2, 2), (:i, :k))
            t_ilm = Tensor(zeros(2, 2, 2), (:i, :l, :m))
            t_lm = Tensor(zeros(2, 2), (:l, :m))
            tn = TensorNetwork([t_ij, t_ik, t_ilm, t_lm])

            old_tensor = t_lm

            @test_throws ArgumentError begin
                new_tensor = Tensor(rand(2, 2), (:a, :b))
                replace!(tn, old_tensor => new_tensor)
            end

            new_tensor = Tensor(rand(2, 2), (:l, :m))
            replace!(tn, old_tensor => new_tensor)

            @test new_tensor === only(filter(t -> issetequal(inds(t), [:l, :m]), tensors(tn)))

            # Check if connections are maintained
            # for label in inds(new_tensor)
            #     index = tn.inds[label]
            #     @test new_tensor in index.links
            #     @test !(old_tensor in index.links)
            # end

            # New tensor network with two tensors with the same inds
            # A = Tensor(rand(2, 2), (:u, :w))
            # B = Tensor(rand(2, 2), (:u, :w))
            # tn = TensorNetwork([A, B])

            # new_tensor = Tensor(rand(2, 2), (:u, :w))

            # replace!(tn, B => new_tensor)
            # @test A === tensors(tn)[1]
            # @test new_tensor === tensors(tn)[2]

            # tn = TensorNetwork([A, B])
            # replace!(tn, A => new_tensor)

            # @test issetequal(tensors(tn), [new_tensor, B])

            # # Test chain of replacements
            # A = Tensor(zeros(2, 2), (:i, :j))
            # B = Tensor(zeros(2, 2), (:j, :k))
            # C = Tensor(zeros(2, 2), (:k, :l))
            # tn = TensorNetwork([A, B, C])

            # @test_throws ArgumentError replace!(tn, A => B, B => C, C => A)

            # new_tensor = Tensor(rand(2, 2), (:i, :j))
            # new_tensor2 = Tensor(ones(2, 2), (:i, :j))

            # replace!(tn, A => new_tensor, new_tensor => new_tensor2)
            # @test issetequal(tensors(tn), [new_tensor2, B, C])
        end
    end

    @testset "contract" begin
        @testset "hyperindex" begin
            let tn = TensorNetwork([Tensor(ones(2, 2), [:a, :i]), Tensor(ones(2), [:i]), Tensor(ones(2, 2), [:b, :i])])
                tn_transformed = transform(tn, Tenet.HyperFlatten())

                result = contract!(tn, :i)
                @test issetequal(inds(result), [:a, :b])

                @test contract(tn_transformed) ≈ only(tensors(result))
            end

            let tn = TensorNetwork([
                    Tensor(ones(2, 2), [:a, :X]),
                    Tensor(ones(2), [:X]),
                    Tensor(ones(2, 2, 2), [:X, :c, :Y]),
                    Tensor(ones(2), [:Y]),
                    Tensor(ones(2, 2, 2), [:Y, :d, :Z]),
                    Tensor(ones(2), [:Z]),
                    Tensor(ones(2, 2, 2), [:Z, :e, :T]),
                    Tensor(ones(2), [:T]),
                    Tensor(ones(2, 2), [:b, :T]),
                ])
                tn_transformed = transform(tn, Tenet.HyperFlatten())

                @test contract(tn) ≈ contract(tn_transformed)
            end
        end
    end
end

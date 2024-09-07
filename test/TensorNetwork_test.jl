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

    @testset "Base.similar" begin
        listtensors = [Tensor(rand(2, 3, 4), (:i, :j, :k)), Tensor(rand(4, 3, 2), (:l, :j, :m))]
        tn = TensorNetwork(listtensors)

        similartn = similar(tn)

        @test length(tensors(tn)) == length(tensors(similartn))
        @test issetequal(inds(tn), inds(similartn))
        @test all(splat(==), zip(inds.(tensors(tn)), inds.(tensors(similartn))))
    end

    @testset "Base.zero" begin
        listtensors = [Tensor(rand(2, 3, 4), (:i, :j, :k)), Tensor(rand(4, 3, 2), (:l, :j, :m))]
        tn = TensorNetwork(listtensors)

        zerostn = zero(tn)

        @test all(tensors(zerostn)) do tns
            iszero(tns)
        end
    end

    @testset "Base.isapprox" begin
        ta = [Tensor([1.0001 2.0001 3.0001; 3.9999 4.9999 5.9999], (:i, :j))]
        tb = [Tensor([1 2 3; 4 5 6], (:i, :j))]
        tn = TensorNetwork(ta)
        approxtn = TensorNetwork(tb)
        atoltrue = 1e-3
        atolfalse = 1e-5

        @test Base.isapprox(tn, approxtn; atol=atoltrue)
        @test !Base.isapprox(tn, approxtn; atol=atolfalse)
    end

    @testset "arrays" begin
        arr1 = rand(2, 3, 4)
        arr2 = rand(4, 3, 2)
        listtensors = [Tensor(arr1, (:i, :j, :k)), Tensor(arr2, (:l, :j, :m))]
        tn = TensorNetwork(listtensors)

        listarrays = arrays(tn)

        @test all(isa.(listarrays, Array))
        @test all([all(tns .== arr) for (tns, arr) in zip(listtensors, listarrays)])
    end

    @testset "Base.eltype" begin
        datatype = ComplexF64
        listtensors = [Tensor(rand(datatype, 2, 3, 4), (:i, :j, :k)), Tensor(rand(datatype, 4, 3, 2), (:l, :j, :m))]
        tn = TensorNetwork(listtensors)

        @test eltype(tn) == datatype
    end

    @testset "neighbors" begin
        t1 = Tensor(rand(2, 3, 4), (:i, :j, :k))
        t2 = Tensor(rand(4, 3, 2), (:l, :j, :m))
        t3 = Tensor(rand(4, 4, 4), (:l, :n, :o))
        listtensors = [t1, t2, t3]
        tn = TensorNetwork(listtensors)

        @testset "by tensor" begin
            @test issetequal(neighbors(tn, t1), [t2])
            @test issetequal(neighbors(tn, t2), [t1, t3])
            @test issetequal(neighbors(tn, t3), [t2])
            @test_throws AssertionError neighbors(tn, Tensor(rand(4, 4, 4), (:a, :b, :c)))
        end

        @testset "by symbol" begin
            @test issetequal(neighbors(tn, :i), inds(t1))
            @test issetequal(neighbors(tn, :j), inds(t1) ∪ inds(t2))
            @test issetequal(neighbors(tn, :l), inds(t2) ∪ inds(t3))
            @test issetequal(neighbors(tn, :n), inds(t3))
            @test_throws AssertionError neighbors(tn, :p)
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
        @testset "two tn" begin
            tensor = Tensor(zeros(2, 3), (:i, :j))
            A = TensorNetwork([tensor])
            B = TensorNetwork()

            merge!(A, B)
            @test only(tensors(A)) === tensor
        end
        @testset "three tn" begin
            tensor1 = Tensor(zeros(2, 3), (:i, :j))
            tensor2 = Tensor(ones(3, 4), (:j, :k))
            tensor3 = Tensor(rand(4, 5), (:k, :l))
            A = TensorNetwork([tensor1])
            B = TensorNetwork([tensor2])
            C = TensorNetwork([tensor3])

            merge!(A, B, C)
            @test length(tensors(A)) == 3
            @test issetequal(inds(A), [:i, :j, :k, :l])
            @test tensor1 ∈ tensors(A)
            @test tensor2 ∈ tensors(A)
            @test tensor3 ∈ tensors(A)
        end
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
        @testset "default kwargs" begin
            tn = rand(TensorNetwork, 10, 3)
            @test tn isa TensorNetwork
            @test length(tensors(tn)) == 10
        end
        @testset "global index" begin
            tn = rand(TensorNetwork, 10, 3; globalind=true)
            @test length(intersect(inds.(tensors(tn))...)) == 1
        end
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

        @test issetequal(tensors(tn; intersects=:i), (t_ij, t_ik, t_ilm))
        @test issetequal(tensors(tn; intersects=:j), (t_ij,))
        @test issetequal(tensors(tn; intersects=:k), (t_ik,))
        @test issetequal(tensors(tn; intersects=:l), (t_ilm, t_lm))
        @test issetequal(tensors(tn; intersects=:m), (t_ilm, t_lm))
        @test issetequal(tensors(tn; contains=(:i, :j)), (t_ij,))
        @test issetequal(tensors(tn; contains=(:i, :k)), (t_ik,))
        @test issetequal(tensors(tn; contains=(:i, :l)), (t_ilm,))
        @test issetequal(tensors(tn; contains=(:l, :m)), (t_ilm, t_lm))
        @test_throws KeyError tensors(tn, intersects=:_)
        @test isempty(tensors(tn; contains=(:j, :l)))
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
        @testset "by einexpr" begin
            tn = rand(TensorNetwork, 5, 3)
            @test contract(tn) isa Tensor

            A = Tensor(rand(2, 2, 2), (:i, :j, :k))
            B = Tensor(rand(2, 2, 2), (:k, :l, :m))
            tn = TensorNetwork([A, B])

            ctn = contract(tn)
            @test ctn isa Tensor
            @test issetequal([:i, :j, :l, :m], inds(ctn))
        end

        @testset "by index" begin
            A = Tensor(rand(2, 2, 2), (:i, :j, :k))
            B = Tensor(rand(2, 2, 2, 2), (:k, :l, :m, :n))
            C = Tensor(rand(2, 2, 2), (:n, :o, :p))
            tn = TensorNetwork([A, B, C])

            ctn = contract(tn, :k)
            @test ctn isa TensorNetwork
            @test length(tensors(ctn)) == 2
            @test issetequal([:i, :j, :l, :m, :n, :o, :p], inds(ctn))
        end

        @testset "by tensor" begin
            A = Tensor(rand(2, 2, 2), (:i, :j, :k))
            B = Tensor(rand(2, 2, 2, 2), (:k, :l, :m, :n))
            newtensor = Tensor(rand(2, 2, 2), (:n, :o, :p))
            tn = TensorNetwork([A, B])

            ctn = contract(tn, newtensor)
            @test tn isa TensorNetwork
            @test issetequal([A, B], tensors(tn))
            @test ctn isa Tensor
            @test issetequal([:i, :j, :l, :m, :o, :p], inds(ctn))
        end

        @testset "hyperindex" begin
            let tn = TensorNetwork([Tensor(ones(2, 2), [:a, :i]), Tensor(ones(2), [:i]), Tensor(ones(2, 2), [:b, :i])])
                tn_transformed = transform(tn, Tenet.HyperFlatten())

                result = contract(tn, :i)
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

    @testset "contract!" begin
        @testset "by index" begin
            A = Tensor(rand(2, 2, 2), (:i, :j, :k))
            B = Tensor(rand(2, 2, 2, 2), (:k, :l, :m, :n))
            C = Tensor(rand(2, 2, 2), (:n, :o, :p))
            tn = TensorNetwork([A, B, C])

            contract!(tn, :k)
            @test tn isa TensorNetwork
            @test length(tensors(tn)) == 2
            @test issetequal([:i, :j, :l, :m, :n, :o, :p], inds(tn))
        end

        @testset "by tensor" begin
            A = Tensor(rand(2, 2, 2), (:i, :j, :k))
            B = Tensor(rand(2, 2, 2, 2), (:k, :l, :m, :n))
            newtensor = Tensor(rand(2, 2, 2), (:n, :o, :p))
            tn = TensorNetwork([A, B])

            ctn = contract!(tn, newtensor)
            @test tn isa TensorNetwork
            @test issetequal([:i, :j, :k, :l, :m, :n, :o, :p], inds(tn))
            @test ctn isa Tensor
            @test issetequal([:i, :j, :l, :m, :o, :p], inds(ctn))
        end
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

            @test only(tensors(tn; contains=(:u, :v))) == replace(t_ij, mapping...)
            @test only(tensors(tn; contains=(:u, :w))) == replace(t_ik, mapping...)
            @test only(tensors(tn; contains=(:u, :x, :y))) == replace(t_ilm, mapping...)
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

        @testset "replace tensors by tensor network" begin
            t_ij = Tensor(zeros(2, 2), (:i, :j))
            t_ik = Tensor(zeros(2, 2), (:i, :k))
            t_ilm = Tensor(zeros(2, 2, 2), (:i, :l, :m))
            t_iln = Tensor(zeros(2, 2, 2), (:i, :l, :n))
            t_mn = Tensor(zeros(2, 2), (:m, :n))
            t_replaced = t_ilm

            tensorsA = [t_ij, t_ik, t_ilm]
            tnA = TensorNetwork(tensorsA)
            tensorsB = [t_iln, t_mn]
            tnB = TensorNetwork(tensorsB)

            tn_nomatch = TensorNetwork([t_iln])
            @test_throws ArgumentError replace!(tnA, t_replaced => tn_nomatch)

            finaltensors = [t_ij, t_ik, t_iln, t_mn]
            finalinds = [:i, :j, :k, :l, :m, :n]

            replace!(tnA, t_replaced => tnB)
            @test length(tensors(tnA)) == (length(tensorsA) + length(tensorsB) - 1)
            @test issetequal(finalinds, inds(tnA))
            @test issetequal(finaltensors, tensors(tnA))
        end

        @testset "overlapping replacement" begin
            A = Tensor(rand(2, 2, 2), (:F, :A, :K))
            B = Tensor(rand(2, 2, 2, 2), (:K, :G, :B, :L))
            C = Tensor(rand(2, 2, 2, 2), (:L, :H, :C, :M))
            D = Tensor(rand(2, 2, 2, 2), (:M, :I, :D, :N))
            E = Tensor(rand(2, 2, 2), (:N, :J, :E))

            old_new = [
                :N => :N,
                :F => :A,
                :M => :O,
                :A => :P,
                :D => :J,
                :B => :K,
                :I => :D,
                :H => :C,
                :G => :B,
                :J => :E,
                :K => :L,
                :L => :U,
                :E => :M,
                :C => :V,
            ]
            tn = TensorNetwork([A, B, C, D, E])

            replace!(tn, old_new...)

            @test issetequal(
                inds.(tensors(tn)), [[:A, :P, :L], [:L, :B, :K, :U], [:U, :C, :V, :O], [:O, :D, :J, :N], [:N, :E, :M]]
            )
        end
    end

    @testset "Base.in" begin
        @testset "by tensor" begin
            tensor1 = Tensor(rand(3, 4), (:i, :j))
            tensor2 = Tensor(rand(4, 5), (:j, :k))
            tn = TensorNetwork([tensor1, tensor2])

            # Broadcast not working
            @test all(∈(tn), [tensor1, tensor2])
        end
        @testset "by symbol" begin
            indices = (:i, :j)
            tn = TensorNetwork([Tensor(rand(3, 4), indices)])

            # Broadcast not working
            @test all(∈(tn), indices)
        end
    end

    @testset "groupinds!" begin
        tn = TensorNetwork([Tensor(zeros(2, 2), [:i, :j]), Tensor(zeros(2, 2), [:i, :j])])
        groupinds!(tn, :i)
        @test inds(tn) == [:i]
        @test size(tn, :i) == 4
        @test Tenet.ntensors(tn) == 2

        tn = TensorNetwork([Tensor(zeros(2, 2), [:i, :j]), Tensor(zeros(2, 2), [:i, :j, :k])])
        groupinds!(tn, :i)
        @test inds(tn) == [:i, :k]
        @test size(tn, :i) == 4
        @test size(tn, :k) == 2
        @test Tenet.ntensors(tn) == 2

        tn = TensorNetwork([Tensor(zeros(2, 2), [:i, :j]), Tensor(zeros(2, 2), [:i, :j]), Tensor(zeros(2), [:j])])
        groupinds!(tn, :i)
        @test inds(tn) == [:i, :j]
        @test size(tn, :i) == 2
        @test size(tn, :j) == 2
        @test Tenet.ntensors(tn) == 2
    end

    @testset "selectdim" begin
        tensor1 = Tensor(rand(3, 4), (:i, :j))
        tensor2 = Tensor(rand(4, 5), (:j, :k))
        tn = TensorNetwork([tensor1, tensor2])
        projdim = 1

        projopentn = selectdim(tn, :i, projdim)
        @test tensors(projopentn) == [Tensor(tensor1[projdim, :], [:j]), tensor2]
        @test issetequal(inds(projopentn), [:j, :k])

        projvirttn = selectdim(tn, :j, projdim)
        @test tensors(projvirttn) == [Tensor(tensor1[:, projdim], [:i]), Tensor(tensor2[projdim, :], [:k])]
        @test issetequal(inds(projvirttn), [:i, :k])
    end

    @testset "Base.conj!" begin
        @testset "for complex" begin
            tensor1 = Tensor(rand(ComplexF64, 3, 4), (:i, :j))
            tensor2 = Tensor(rand(ComplexF64, 4, 5), (:j, :k))
            complextn = TensorNetwork([tensor1, tensor2])

            @test -imag.(tensors(complextn)) == imag.(tensors(conj!(complextn)))
        end
        @testset "for real" begin
            tensor1 = Tensor(rand(3, 4), (:i, :j))
            tensor2 = Tensor(rand(4, 5), (:j, :k))
            realtn = TensorNetwork([tensor1, tensor2])

            @test tensors(conj!(realtn)) == tensors(realtn)
        end
    end

    @testset "@unsafe_region" begin
        @testset "safe region" begin
            tn = TensorNetwork([Tensor(ones(2, 2), [:a, :b]), Tensor(ones(2, 2), [:b, :c])])
            Tenet.@unsafe_region tn begin
                tensor = Tensor(ones(2, 2), [:c, :d])
                push!(tn, tensor)
                @test length(tensors(tn)) == 3
            end
            @test length(tensors(tn)) == 3
        end

        @testset "unsafe region" begin
            tn = TensorNetwork([Tensor(ones(2, 2), [:a, :b]), Tensor(ones(2, 2), [:b, :c])])
            tn_copy = copy(tn)
            @test_throws DimensionMismatch Tenet.@unsafe_region tn begin
                tensor = Tensor(ones(3, 2), [:c, :d])
                push!(tn, tensor)
            end
            @test tn == tn_copy

            Tenet.@unsafe_region tn begin
                tensor = Tensor(ones(3, 2), [:c, :d])
                push!(tn, tensor)
                @test length(tensors(tn)) == 3
                pop!(tn, tensor)
            end

            @testset "SubArray" begin
                a = Tensor(view(ones(2, 2), 1:2, 1:2), [:a, :b])
                b = Tensor(view(ones(2, 2), 1:2, 1:2), [:b, :c])
                c = Tensor(view(ones(3, 2), 1:3, 1:2), [:c, :d])
                tn = TensorNetwork([a, b])

                @test_throws DimensionMismatch Tenet.@unsafe_region tn begin
                    push!(tn, c)
                end

                @test tensors(tn)[1] === a
                @test tensors(tn)[2] === b
            end
        end
    end

    @testset "LinearAlgebra.svd!" begin
        M = rand(ComplexF64, 4, 3)
        left_inds = [:i]
        right_inds = [:j]
        indsM = left_inds ∪ right_inds

        U, S, V = svd(M)
        ctn = TensorNetwork([Tensor(M, indsM)])

        svd!(ctn; left_inds, right_inds)
        @test isapprox([S, U, conj(V)], tensors(ctn); rtol=1e-9)
        @test isapprox(permutedims(contract(ctn), indsM), M; rtol=1e-9)
    end

    @testset "LinearAlgebra.qr!" begin
        M = rand(ComplexF64, 4, 3)
        left_inds = [:i]
        right_inds = [:j]
        indsM = left_inds ∪ right_inds

        F = qr(M)
        ctn = TensorNetwork([Tensor(M, indsM)])

        qr!(ctn; left_inds, right_inds)
        @test isapprox([F.R, Matrix(F.Q)], tensors(ctn); rtol=1e-9)
        @test isapprox(permutedims(contract(ctn), indsM), M; rtol=1e-9)
    end

    @testset "LinearAlgebra.lu!" begin
        M = rand(ComplexF64, 4, 3)
        left_inds = [:i]
        right_inds = [:j]
        indsM = left_inds ∪ right_inds
        tensor = Tensor(M, indsM)
        ctn = TensorNetwork([tensor])

        L, U, P = lu(tensor; left_inds, right_inds)
        lu!(ctn; left_inds, right_inds)

        @test issetequal(Set(parent.([P, L, U])), Set(arrays(ctn)))
        @test isapprox(permutedims(contract(ctn), indsM), M; rtol=1e-9)
    end
end

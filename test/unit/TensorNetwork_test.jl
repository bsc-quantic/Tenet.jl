using Test
using Tenet
using Tenet: hasinterface, TensorNetworkInterface
using Serialization
using Graphs: neighbors
using LinearAlgebra

@testset "interface" begin
    @test hasinterface(TensorNetworkInterface(), TensorNetwork)
end

@testset "constructors" begin
    @testset "empty" begin
        tn = TensorNetwork()
        @test isempty(tensors(tn))
        @test isempty(inds(tn))
        @test isempty(size(tn))
    end

    @testset "list of tensors" begin
        tensor = Tensor(zeros(2, 3), (:i, :j))
        tn = TensorNetwork([tensor])

        @test only(tensors(tn)) === tensor
        @test issetequal(inds(tn), [:i, :j])
        @test size(tn) == Dict(:i => 2, :j => 3)
        @test issetequal(inds(tn; set=:open), [:i, :j])
        @test isempty(inds(tn; set=:hyper))
    end

    @testset "rand" begin
        @testset "default kwargs" begin
            tn = rand(TensorNetwork, 10, 3)
            @test tn isa TensorNetwork
            @test ntensors(tn) == 10
        end
        @testset "global index" begin
            tn = rand(TensorNetwork, 10, 3; globalind=true)
            @test length(intersect(inds.(tensors(tn))...)) == 1
        end
    end

    @testset "throw: indices of different dimensions" begin
        tensor1 = Tensor(zeros(2, 2), (:i, :j))
        tensor2 = Tensor(zeros(3, 3), (:j, :k))
        @test_throws DimensionMismatch tn = TensorNetwork([tensor1, tensor2])
    end
end

@testset "inds" begin
    tn = TensorNetwork([
        Tensor(zeros(2, 2), (:i, :j)),
        Tensor(zeros(2, 2), (:i, :k)),
        Tensor(zeros(2, 2, 2), (:i, :l, :m)),
        Tensor(zeros(2, 2), (:l, :m)),
    ])

    @testset "`inds` returns a list of the indices" begin
        @test issetequal(inds(tn), [:i, :j, :k, :l, :m])
        @test ninds(tn) == 5

        @testset "`inds(; set = :all)` is equal to naive `inds`" begin
            @test inds(tn; set=:all) == inds(tn)
            @test ninds(tn; set=:all) == ninds(tn)
        end
    end

    @testset "`inds(; set = :open)` returns a list of indices" begin
        @test issetequal(inds(tn; set=:open), [:j, :k])
        @test ninds(tn; set=:open) == 2
    end

    @testset "`inds(; set = :inner)` returns a list of indices" begin
        @test issetequal(inds(tn; set=:inner), [:i, :l, :m])
        @test ninds(tn; set=:inner) == 3
    end

    @testset "`inds(; set = :hyper)` returns a list of indices" begin
        @test issetequal(inds(tn; set=:hyper), [:i])
        @test ninds(tn; set=:hyper) == 1

        @testset let
            tn = TensorNetwork([Tensor(zeros(2), (:i,)), Tensor(zeros(2), (:i,)), Tensor(zeros(2), (:i,))])
            @test issetequal(inds(tn; set=:hyper), [:i])
            @test ninds(tn; set=:hyper) == 1
        end

        @testset let
            tensor = Tensor(zeros(2, 2, 2), (:i, :i, :i))
            tn = TensorNetwork([tensor])
            @test issetequal(inds(tn; set=:hyper), [:i])
            @test ninds(tn; set=:hyper) == 1
        end

        @testset let
            tensor = Tensor(zeros(2, 2, 2), (:i, :i, :i))
            tn = TensorNetwork()
            push!(tn, tensor)
            @test issetequal(inds(tn; set=:hyper), [:i])
            @test ninds(tn; set=:hyper) == 1
        end
    end

    @testset "`inds(; parallelto)` returns the indices parallel to the given one" begin
        @testset let
            tn = TensorNetwork([Tensor(zeros(2, 2), [:i, :j]), Tensor(zeros(2, 2), [:i, :j])])
            @test issetequal(inds(tn; parallelto=:i), [:j])
            @test ninds(tn; parallelto=:i) == 1
        end

        @testset let
            tn = TensorNetwork([Tensor(zeros(2, 2), [:i, :j]), Tensor(zeros(2, 2, 2), [:i, :j, :k])])
            @test issetequal(inds(tn; parallelto=:i), [:j])
            @test ninds(tn; parallelto=:i) == 1
        end

        @testset let
            tn = TensorNetwork([Tensor(zeros(2, 2), [:i, :j]), Tensor(zeros(2, 2), [:i, :j]), Tensor(zeros(2), [:j])])
            @test isempty(inds(tn; parallelto=:i))
            @test ninds(tn; parallelto=:i) == 0
        end
    end
end

@testset "tensors" begin
    t_ij = Tensor(zeros(2, 2), (:i, :j))
    t_ik = Tensor(zeros(2, 2), (:i, :k))
    t_ilm = Tensor(zeros(2, 2, 2), (:i, :l, :m))
    t_lm = Tensor(zeros(2, 2), (:l, :m))
    tn = TensorNetwork([t_ij, t_ik, t_ilm, t_lm])

    @testset "all tensors" begin
        @test issetequal(tensors(tn), [t_ij, t_ik, t_ilm, t_lm])
        @test ntensors(tn) == 4
    end

    @testset "`tensors(; contains = i)` returns a list of tensors containing index `i`" begin
        @test issetequal(tensors(tn; contains=(:i, :j)), (t_ij,))
        @test ntensors(tn; contains=(:i, :j)) == 1

        @test issetequal(tensors(tn; contains=(:i, :k)), (t_ik,))
        @test ntensors(tn; contains=(:i, :k)) == 1

        @test issetequal(tensors(tn; contains=(:i, :l)), (t_ilm,))
        @test ntensors(tn; contains=(:i, :l)) == 1

        @test issetequal(tensors(tn; contains=(:l, :m)), (t_ilm, t_lm))
        @test ntensors(tn; contains=(:l, :m)) == 2

        @test isempty(tensors(tn; contains=(:j, :l)))
        @test ntensors(tn; contains=(:j, :l)) == 0
    end

    @testset "`tensors(; intersects = i)` returns a list of tensors intersecting index `i`" begin
        @test issetequal(tensors(tn; intersects=:i), (t_ij, t_ik, t_ilm))
        @test ntensors(tn; intersects=:i) == 3

        @test issetequal(tensors(tn; intersects=:j), (t_ij,))
        @test ntensors(tn; intersects=:j) == 1

        @test issetequal(tensors(tn; intersects=:k), (t_ik,))
        @test ntensors(tn; intersects=:k) == 1

        @test issetequal(tensors(tn; intersects=:l), (t_ilm, t_lm))
        @test ntensors(tn; intersects=:l) == 2

        @test issetequal(tensors(tn; intersects=:m), (t_ilm, t_lm))
        @test ntensors(tn; intersects=:m) == 2

        @test issetequal(tensors(tn; intersects=(:i, :m)), (t_ij, t_ik, t_ilm, t_lm))
        @test ntensors(tn; intersects=(:i, :m)) == 4

        @test isempty(tensors(tn; intersects=:_))
        @test ntensors(tn; intersects=:_) == 0
    end
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

@testset "copy" begin
    @testset let
        tensor = Tensor(zeros(2, 2), (:i, :j))
        tn = TensorNetwork([tensor])
        tn_copy = copy(tn)

        @test tensors(tn_copy) !== tensors(tn) && all(tensors(tn_copy) .=== tensors(tn))
        @test inds(tn) !== inds(tn_copy) && issetequal(inds(tn), inds(tn_copy))
    end

    @testset "case: tensor of subarray" begin
        tensor = Tensor(rand(2, 2, 2), (:i, :j, :k))
        subtensor = view(tensor, :i => 1)
        tn = TensorNetwork([tensor])
        tn_copy = copy(tn)

        # it's important that we don't return the same cached sorted vector...
        @test tensors(tn_copy) !== tensors(tn)
        @test inds(tn) !== inds(tn_copy)

        # ...but that the tensors/inds are the same
        @test all(tensors(tn_copy) .=== tensors(tn))
        @test issetequal(inds(tn), inds(tn_copy))
    end
end

@testset "hastensor" begin
    tensor = Tensor(zeros(2, 2), (:i, :j))
    tn = TensorNetwork([tensor])

    @test hastensor(tn, tensor)

    copied_tensor = copy(tensor)
    @test !hastensor(tn, copied_tensor)

    copied_tn = copy(tn)
    @test hastensor(copied_tn, tensor)

    @testset "Base.in alias" begin
        tensor1 = Tensor(rand(3, 4), (:i, :j))
        tensor2 = Tensor(rand(4, 5), (:j, :k))
        tn = TensorNetwork([tensor1, tensor2])

        # Broadcast not working
        @test all(∈(tn), [tensor1, tensor2])
    end
end

@testset "hasind" begin
    tn = TensorNetwork([Tensor(zeros(2, 2), (:i, :j))])

    @test hasind(tn, :i)
    @test hasind(tn, :j)
    @test !hasind(tn, :k)

    @testset "Base.in alias" begin
        @test :i ∈ tn
    end
end

@testset "size" begin
    tn = TensorNetwork([
        Tensor(zeros(2, 3), (:i, :j)),
        Tensor(zeros(2, 4), (:i, :k)),
        Tensor(zeros(2, 5, 6), (:i, :l, :m)),
        Tensor(zeros(5, 6), (:l, :m)),
    ])

    @test size(tn) == Dict([:i => 2, :j => 3, :k => 4, :l => 5, :m => 6])
    @test all([size(tn, :i) == 2, size(tn, :j) == 3, size(tn, :k) == 4, size(tn, :l) == 5, size(tn, :m) == 6])
end

# mutating methods
@testset "push!" begin
    tn = TensorNetwork()
    tensor = Tensor(zeros(2, 2, 2), (:i, :j, :k))
    push!(tn, tensor)

    @test ntensors(tn) == 1
    @test issetequal(inds(tn), [:i, :j, :k])
    @test size(tn) == Dict(:i => 2, :j => 2, :k => 2)
    @test issetequal(inds(tn; set=:open), [:i, :j, :k])
    @test isempty(inds(tn; set=:hyper))

    @testset let tn = copy(tn)
        @test_throws DimensionMismatch push!(tn, Tensor(zeros(3, 3), (:i, :j)))
    end

    @testset let tn = copy(tn)
        @test_throws DimensionMismatch begin
            tensor = Tensor(zeros(2, 3), (:i, :i))
            push!(tn, tensor)
        end
    end

    # TODO test `PushEffect` is handled
end

@testset "append!" begin
    @testset "empty tn and list of tensor" begin
        tensor = Tensor(zeros(2, 3), (:i, :j))
        A = TensorNetwork()

        append!(A, [tensor])
        @test only(tensors(A)) === tensor
    end

    @testset "tn and empty tn" begin
        tensor = Tensor(zeros(2, 3), (:i, :j))
        A = TensorNetwork([tensor])
        B = TensorNetwork()

        append!(A, B)
        @test only(tensors(A)) === tensor
    end

    @testset "three tn" begin
        tensor1 = Tensor(zeros(2, 3), (:i, :j))
        tensor2 = Tensor(ones(3, 4), (:j, :k))
        tensor3 = Tensor(rand(4, 5), (:k, :l))
        A = TensorNetwork([tensor1])
        B = TensorNetwork([tensor2])
        C = TensorNetwork([tensor3])

        append!(A, B)
        append!(A, C)
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
        @test ntensors(tn) == 0
        @test isempty(tensors(tn))
        @test isempty(size(tn))
    end

    @testset "by symbol" begin
        tensor = Tensor(zeros(2, 3), (:i, :j))
        tn = TensorNetwork([tensor])

        @test only(pop!(tn, :i)) === tensor
        @test ntensors(tn) == 0
        @test isempty(tensors(tn))
        @test isempty(size(tn))
    end

    @testset "by symbols" begin
        A = Tensor(zeros(2, 3), (:i, :j))
        B = Tensor(zeros(3, 2), (:j, :k))
        tn = TensorNetwork([A, B])

        @test only(pop!(tn, (:i, :j))) === A
        @test ntensors(tn) == 1
        @test size(tn) == Dict(:j => 3, :k => 2)
    end
end

@testset "delete!" begin
    tensor = Tensor(zeros(2, 3), (:i, :j))
    tn = TensorNetwork([tensor])

    @test delete!(tn, tensor) === tn
    @test ntensors(tn) == 0
    @test isempty(tensors(tn))
    @test isempty(size(tn))

    # TODO test `DeleteEffect` is handled
end

@testset "Base.replace!" begin
    # TODO test `ReplaceEffect` is handled

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
        @testset "basic replacement" begin
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
            for ind in inds(new_tensor)
                tensors_with_ind = tn.indexmap[ind]
                @test new_tensor ∈ tensors_with_ind
                @test !(old_tensor ∈ tensors_with_ind)
            end
        end

        @testset "`TensorNetwork` with tensors of equal indices" begin
            A = Tensor(rand(2, 2), (:u, :w))
            B = Tensor(rand(2, 2), (:u, :w))
            tn = TensorNetwork([A, B])

            new_tensor = Tensor(rand(2, 2), (:u, :w))

            replace!(tn, B => new_tensor)
            @test A ∈ tensors(tn)
            @test new_tensor ∈ tensors(tn)

            tn = TensorNetwork([A, B])
            replace!(tn, A => new_tensor)

            @test issetequal(tensors(tn), [new_tensor, B])
        end

        @testset "sequence of replacements" begin
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

        @testset "replace with itself" begin
            A = Tensor(rand(2, 2), (:i, :j))
            B = Tensor(rand(2, 2), (:j, :k))
            C = Tensor(rand(2, 2), (:k, :l))
            tn = TensorNetwork([A, B, C])

            replace!(tn, A => A)

            @test issetequal(tensors(tn), [A, B, C])
        end
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
            inds.(tensors(tn)), [(:A, :P, :L), (:L, :B, :K, :U), (:U, :C, :V, :O), (:O, :D, :J, :N), (:N, :E, :M)]
        )
    end
end

# derived methods
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

@testset "selectdim" begin
    tn = TensorNetwork([Tensor(zeros(2, 2), (:i, :j))])

    tn_selected = selectdim(tn, :i, 1)
    @test !hasind(tn_selected, :i)

    tn_selected = selectdim(tn, :i, 1:1)
    @test hasind(tn_selected, :i)
    @test size(tn_selected, :i) == 1
end

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

    @testset "hyperindex" begin
        let
            tn = TensorNetwork([Tensor(ones(2, 2), [:a, :i]), Tensor(ones(2), [:i]), Tensor(ones(2, 2), [:b, :i])])
            tn_transformed = transform(tn, Tenet.HyperFlatten())

            result = contract(tn, :i)
            @test issetequal(inds(result), [:a, :b])

            @test contract(tn_transformed) ≈ only(tensors(result))
        end

        let
            tn = TensorNetwork([
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
        @test ntensors(tn) == 2
        @test !hasind(tn, :k)
        @test issetequal([:i, :j, :l, :m, :n, :o, :p], inds(tn))
    end
end

# derived methods
@testset "Base.similar" begin
    listtensors = [Tensor(rand(2, 3, 4), (:i, :j, :k)), Tensor(rand(4, 3, 2), (:l, :j, :m))]
    tn = TensorNetwork(listtensors)

    similartn = similar(tn)

    @test ntensors(tn) == length(tensors(similartn))
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
        @test_throws ArgumentError neighbors(tn, Tensor(rand(4, 4, 4), (:a, :b, :c)))
    end

    @testset "by symbol" begin
        @test issetequal(neighbors(tn, :i), inds(t1))
        @test issetequal(neighbors(tn, :j), inds(t1) ∪ inds(t2))
        @test issetequal(neighbors(tn, :l), inds(t2) ∪ inds(t3))
        @test issetequal(neighbors(tn, :n), inds(t3))
        @test_throws ArgumentError neighbors(tn, :p)
    end
end

@testset "fuse!" begin
    tn = TensorNetwork([Tensor(zeros(2, 2), [:i, :j]), Tensor(zeros(2, 2), [:i, :j])])
    fuse!(tn, :i)
    @test inds(tn) == [:i]
    @test size(tn, :i) == 4
    @test Tenet.ntensors(tn) == 2

    tn = TensorNetwork([Tensor(zeros(2, 2), [:i, :j]), Tensor(zeros(2, 2, 2), [:i, :j, :k])])
    fuse!(tn, :i)
    @test issetequal(inds(tn), [:i, :k])
    @test size(tn, :i) == 4
    @test size(tn, :k) == 2
    @test Tenet.ntensors(tn) == 2

    tn = TensorNetwork([Tensor(zeros(2, 2), [:i, :j]), Tensor(zeros(2, 2), [:i, :j]), Tensor(zeros(2), [:j])])
    fuse!(tn, :i)
    @test issetequal(inds(tn), [:i, :j])
    @test size(tn, :i) == 2
    @test size(tn, :j) == 2
    @test Tenet.ntensors(tn) == 3
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

@testset "Base.conj" begin
    tensor1 = Tensor(rand(ComplexF64, 3, 4), (:i, :j))
    tensor2 = Tensor(rand(ComplexF64, 4, 5), (:j, :k))
    complextn = TensorNetwork([tensor1, tensor2])

    @test -imag.(tensors(complextn)) == imag.(tensors(conj(complextn)))
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
            @test ntensors(tn) == 3
        end
        @test ntensors(tn) == 3
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
            @test ntensors(tn) == 3
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

        @testset "copy inside unsafe region" begin
            tn = TensorNetwork([Tensor(ones(2, 2), [:a, :b]), Tensor(ones(2, 2), [:b, :c])])

            @test_throws DimensionMismatch Tenet.@unsafe_region tn begin
                tensor = Tensor(ones(3, 2), [:c, :d])
                push!(tn, tensor)
                tn2 = TensorNetwork([Tensor(ones(2, 2), [:a, :b]), Tensor(ones(2, 2), [:b, :c])])
                push!(tn2, tensor) # tn2 is not specified in @unsafe_region argument
                @test ntensors(tn) == 3
                delete!(tn, tensor)
            end

            # Here still errors since at the end `tn2` is inconsistent:
            @test_throws DimensionMismatch Tenet.@unsafe_region tn begin
                tensor = Tensor(ones(3, 2), [:c, :d])
                push!(tn, tensor)
                tn2 = copy(tn)
                push!(tn2, tensor)
                @test ntensors(tn) == 3
                delete!(tn, tensor)
            end

            # Double copy should also throw an error:
            @test_throws DimensionMismatch Tenet.@unsafe_region tn begin
                tensor = Tensor(ones(3, 2), [:c, :d])
                push!(tn, tensor)
                tn2 = copy(tn)
                tn3 = copy(tn2)
                push!(tn3, tensor)
                @test ntensors(tn) == 3
                delete!(tn, tensor)
            end

            Tenet.@unsafe_region tn begin # This should not throw an error
                tensor = Tensor(ones(3, 2), [:c, :d])
                push!(tn, tensor)
                tn2 = copy(tn)
                push!(tn2, tensor)  # tn2 is not specified in @unsafe_region
                @test ntensors(tn) == 3
                delete!(tn, tensor)
                delete!(tn2, tensor)
            end
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
    qr!(ctn; left_inds, right_inds, virtualind=:k)

    # `LinearAlgebra.qr` decomposition is full, but we truncate when matrix is not square in Tenet
    @test F.Q[:, 1:3] ≈ parent(ctn[:i, :k])
    @test F.R ≈ parent(ctn[:k, :j])
    @test M ≈ parent(permutedims(contract(ctn), indsM))
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

@testset "Serialization" begin
    tn = rand(TensorNetwork, 10, 3)

    # Serialize
    buffer = IOBuffer()
    serialize(buffer, tn)
    seekstart(buffer)
    content = read(buffer)

    # Deserialize
    read_buffer = IOBuffer(content)
    tn2 = deserialize(read_buffer)

    @test tn == tn2
end

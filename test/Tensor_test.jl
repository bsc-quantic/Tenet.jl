@testset "Tensor" begin
    @testset "Constructors" begin
        @testset "Number" begin
            tensor = Tensor(1.0, tags = Set(["TEST"]))
            @test labels(tensor) == ()
            @test parent(tensor) == fill(1.0)
            @test hastag(tensor, "TEST")
        end

        @testset "Array" begin
            data = ones(2, 2, 2)
            tensor = Tensor(data, [:i, :j, :k])

            @test labels(tensor) == (:i, :j, :k)
            @test parent(tensor) === data

            @test_throws DimensionMismatch Tensor(zeros(2, 3), (:i, :i))
        end
    end

    @testset "copy" begin
        tensor = Tensor(zeros(2, 2, 2), (:i, :j, :k))
        @test tensor !== copy(tensor)
        @test parent(tensor) === parent(copy(tensor))
        @test labels(tensor) == labels(copy(tensor))
        @test labels(tensor) === labels(copy(tensor))
        @test tensor.meta == copy(tensor).meta
        @test tensor.meta !== copy(tensor).meta

        @test copy(view(tensor, :i => 1)) isa Tensor
        @test parent(copy(view(tensor, :i => 1))) isa Array
    end

    @testset "isequal" begin
        tensor = Tensor(zeros(2, 2, 2), (:i, :j, :k))
        @test tensor == copy(tensor)
        @test tensor != zeros(size(tensor)...)
        @test zeros(size(tensor)...) != tensor

        @test tensor ∈ [tensor]
        @test copy(tensor) ∈ [tensor]
        @test tensor ∈ [copy(tensor)]
        @test zeros(size(tensor)...) ∉ [tensor]

        @test tensor ∈ Set([tensor])
        @test zeros(size(tensor)...) ∉ Set([tensor])

        @test tensor == permutedims(tensor, (3, 1, 2))
    end

    @testset "isapprox" begin
        data = rand(2, 3, 4, 5)
        tensor = Tensor(data, (:i, :j, :k, :l))

        @test tensor ≈ copy(tensor)
        @test tensor ≈ permutedims(tensor, (3, 1, 2, 4))
        @test tensor ≈ permutedims(tensor, (2, 4, 1, 3))
        @test tensor ≈ permutedims(tensor, (4, 3, 2, 1))
        @test tensor ≈ tensor .+ 1e-14

        @test !(tensor ≈ Tensor(data, (:i, :m, :n, :l)))
        @test !(tensor ≈ Tensor(rand(2, 2, 2), (:i, :j, :k)))
    end

    @testset "Base.replace" begin
        # no :alias in meta
        tensor = Tensor(zeros(2, 2, 2), (:i, :j, :k))
        @test labels(replace(tensor, :i => :u, :j => :v, :k => :w)) == (:u, :v, :w)
        @test parent(replace(tensor, :i => :u, :j => :v, :k => :w)) === parent(tensor)

        @test labels(replace(tensor, :a => :u, :b => :v, :c => :w)) == (:i, :j, :k)
        @test parent(replace(tensor, :a => :u, :b => :v, :c => :w)) === parent(tensor)

        # :alias in meta
        tensor = Tensor(zeros(2, 2, 2), (:i, :j, :k); alias = Dict(:left => :i, :right => :j, :up => :k))

        replaced_tensor = replace(tensor, :i => :u, :j => :v, :k => :w)
        @test labels(replaced_tensor) == (:u, :v, :w)
        @test parent(replaced_tensor) === parent(tensor)
        @test replaced_tensor.meta[:alias] == Dict(:left => :u, :right => :v, :up => :w)
    end

    @testset "dim" begin
        tensor = Tensor(zeros(2, 2, 2), (:i, :j, :k))
        @test dim(tensor, 1) == 1
        for (i, label) in enumerate(labels(tensor))
            @test dim(tensor, label) == i
        end

        @test_throws BoundsError dim(tensor, :_)
    end

    @testset "Broadcasting" begin
        data = rand(2, 2, 2)
        @test begin
            tensor = Tensor(data, (:a, :b, :c))
            tensor = tensor .+ one(eltype(tensor))

            parent(tensor) == data .+ one(eltype(tensor))
        end

        @test begin
            tensor = Tensor(data, (:a, :b, :c))
            tensor = sin.(tensor)

            parent(tensor) == sin.(data)
        end
    end

    @testset "selectdim" begin
        data = rand(2, 2, 2)
        tensor = Tensor(data, (:i, :j, :k))

        @test parent(selectdim(tensor, :i, 1)) == selectdim(data, 1, 1)
        @test parent(selectdim(tensor, :j, 2)) == selectdim(data, 2, 2)
        @test issetequal(labels(selectdim(tensor, :i, 1)), (:j, :k))
        @test issetequal(labels(selectdim(tensor, :i, 1:1)), (:i, :j, :k))
    end

    @testset "view" begin
        data = rand(2, 2, 2)
        tensor = Tensor(data, (:i, :j, :k))

        @test parent(view(tensor, 2, :, :)) == view(data, 2, :, :)
        @test parent(view(tensor, :i => 1)) == view(data, 1, :, :)
        @test parent(view(tensor, :j => 2)) == view(data, :, 2, :)
        @test parent(view(tensor, :i => 2, :k => 1)) == view(data, 2, :, 1)
        @test :i ∉ labels(view(tensor, :i => 1))

        @test parent(view(tensor, :i => 1:1)) == view(data, 1:1, :, :)
        @test :i ∈ labels(view(tensor, :i => 1:1))
    end

    @testset "permutedims" begin
        data = rand(2, 2, 2)
        tensor = Tensor(data, (:i, :j, :k))
        perm = (3, 1, 2)

        @test permutedims(tensor, perm) |> labels == (:k, :i, :j)
        @test permutedims(tensor, perm) |> parent == permutedims(data, perm)
        @test permutedims(tensor, perm).meta !== tensor.meta

        newtensor = Tensor(similar(data), (:a, :b, :c))
        permutedims!(newtensor, tensor, perm)
        @test parent(newtensor) == parent(permutedims(tensor, perm))
    end

    @testset "indexing" begin
        data = rand(2, 2, 2)
        tensor = Tensor(data, (:i, :j, :k))

        @test axes(tensor) == axes(data)
        @test first(tensor) == first(data)
        @test last(tensor) == last(data)
        @test tensor[1, :, 2] == data[1, :, 2]
        @test tensor[i = 1, k = 2] == data[1, :, 2]

        tensor[1] = 0
        @test tensor[1] == data[1]

        for i in [0, -1, length(tensor) + 1]
            @test_throws BoundsError tensor[i]
        end
    end

    @testset "iteration" begin
        data = rand(2, 2, 2)
        tensor = Tensor(data, (:i, :j, :k))
        @test all(x -> ==(x...), zip(tensor, data))
    end

    @testset "adjoint" begin
        @testset "Vector" begin
            data = rand(Complex{Float64}, 2)
            tensor = Tensor(data, (:i,); test = "TEST")

            @test adjoint(tensor) |> labels == labels(tensor)
            @test adjoint(tensor) |> ndims == 1
            @test adjoint(tensor).meta == tensor.meta

            @test isapprox(only(tensor' * tensor), data' * data)
        end

        @testset "Matrix" begin
            using LinearAlgebra: tr

            data = rand(Complex{Float64}, 2, 2)
            tensor = Tensor(data, (:i, :j); test = "TEST")

            @test adjoint(tensor) |> labels == labels(tensor)
            @test adjoint(tensor) |> ndims == 2
            @test adjoint(tensor).meta == tensor.meta

            @test isapprox(only(tensor' * tensor), tr(data' * data))
        end
    end

    @testset "transpose" begin
        @testset "Vector" begin
            data = rand(Complex{Float64}, 2)
            tensor = Tensor(data, (:i,); test = "TEST")

            @test transpose(tensor) |> labels == labels(tensor)
            @test transpose(tensor) |> ndims == 1
            @test transpose(tensor).meta == tensor.meta

            @test isapprox(only(transpose(tensor) * tensor), transpose(data) * data)
        end

        @testset "Matrix" begin
            using LinearAlgebra: tr

            data = rand(Complex{Float64}, 2, 2)
            tensor = Tensor(data, (:i, :j); test = "TEST")

            @test transpose(tensor) |> labels == (:j, :i)
            @test transpose(tensor) |> ndims == 2
            @test transpose(tensor).meta == tensor.meta

            @test isapprox(only(transpose(tensor) * tensor), tr(transpose(data) * data))
        end
    end

    @testset "expand" begin
        data = rand(2, 2, 2)
        tensor = Tensor(data, (:i, :j, :k))

        let new = expand(tensor, label = :x, axis = 1)
            @test labels(new) == (:x, :i, :j, :k)
            @test size(new, :x) == 1
            @test selectdim(new, :x, 1) == tensor
        end

        let new = expand(tensor, label = :x, axis = 3)
            @test labels(new) == (:i, :j, :x, :k)
            @test size(new, :x) == 1
            @test selectdim(new, :x, 1) == tensor
        end

        let new = expand(tensor, label = :x, axis = 1, size = 2, method = :zeros)
            @test labels(new) == (:x, :i, :j, :k)
            @test size(new, :x) == 2
            @test selectdim(new, :x, 1) == tensor
            @test selectdim(new, :x, 2) == Tensor(zeros(size(data)...), labels(tensor))
        end

        let new = expand(tensor, label = :x, axis = 1, size = 2, method = :repeat)
            @test labels(new) == (:x, :i, :j, :k)
            @test size(new, :x) == 2
            @test selectdim(new, :x, 1) == tensor
            @test selectdim(new, :x, 2) == tensor
        end
    end
end

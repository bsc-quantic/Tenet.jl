@testset "Tensor" begin
    using Tenet: Tensor, labels

    @testset "Constructor" begin
        tensor = Tensor(zeros(2, 2, 2), (:i, :j, :k))
        @test labels(tensor) == (:i, :j, :k)
        @test labels(Tensor(zeros(2, 2, 2), [:i, :j, :k])) == (:i, :j, :k)
    end

    @testset "copy" begin
        tensor = Tensor(zeros(2, 2, 2), (:i, :j, :k))
        @test tensor !== copy(tensor)
        @test parent(tensor) === parent(copy(tensor))
        @test labels(tensor) == labels(copy(tensor))
        @test labels(tensor) === labels(copy(tensor))
        @test tensor.meta == copy(tensor).meta
        @test tensor.meta !== copy(tensor).meta
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
    end

    @testset "Base.replace" begin
        tensor = Tensor(zeros(2, 2, 2), (:i, :j, :k))
        @test labels(replace(tensor, :i => :u, :j => :v, :k => :w)) == (:u, :v, :w)
        @test parent(replace(tensor, :i => :u, :j => :v, :k => :w)) === parent(tensor)

        @test labels(replace(tensor, :a => :u, :b => :v, :c => :w)) == (:i, :j, :k)
        @test parent(replace(tensor, :a => :u, :b => :v, :c => :w)) === parent(tensor)
    end

    @testset "dim" begin
        using Tenet: dim

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

    @testset "tags" begin
        using Tenet: tags, tag!, untag!, hastag

        tensor = Tensor(zeros(2, 2, 2), (:i, :j, :k), tags = Set{String}(["TAG_A", "TAG_B"]))

        @test issetequal(tags(tensor), ["TAG_A", "TAG_B"])

        tag!(tensor, "TAG_C")
        @test hastag(tensor, "TAG_C")

        untag!(tensor, "TAG_C")
        @test !hastag(tensor, "TAG_C")

        @test untag!(tensor, "TAG_UNEXISTANT") == tags(tensor)
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

    @testset "adjoint" begin
        @testset "Vector" begin
            data = rand(Complex{Float64}, 2)
            tensor = Tensor(data, (:i,); test = "TEST")

            @test adjoint(tensor) |> labels == labels(tensor)
            @test adjoint(tensor) |> ndims == 1
            @test adjoint(tensor).meta == tensor.meta

            @test only(tensor' * tensor) == data' * data
        end

        @testset "Matrix" begin
            using LinearAlgebra: tr

            data = rand(Complex{Float64}, 2, 2)
            tensor = Tensor(data, (:i, :j); test = "TEST")

            @test adjoint(tensor) |> labels == labels(tensor)
            @test adjoint(tensor) |> ndims == 2
            @test adjoint(tensor).meta == tensor.meta

            @test only(tensor' * tensor) == tr(data' * data)
        end
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
end
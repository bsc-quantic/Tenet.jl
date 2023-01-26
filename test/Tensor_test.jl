@testset "Tensor" begin
    using Tenet: Tensor, labels

    @testset "Constructor" begin
        tensor = Tensor(zeros(2, 2, 2), (:i, :j, :k))
        @test labels(tensor) == (:i, :j, :k)
        @test labels(Tensor(zeros(2, 2, 2), [:i, :j, :k])) == (:i, :j, :k)
    end

    @testset "isequal" begin
        tensor = Tensor(zeros(2, 2, 2), (:i, :j, :k))
        @test tensor == copy(tensor)
        @test tensor != zeros(size(tensor)...)

        @test tensor ∈ [tensor]
        @test copy(tensor) ∈ [tensor]
        @test zeros(size(tensor)...) ∉ [tensor]

        @test tensor ∈ [copy(tensor)]
        @test tensor ∈ Set([tensor])
        @test zeros(size(tensor)...) ∉ Set([tensor])
    end

    @testset "reindex" begin
        using Tenet: reindex

        tensor = Tensor(zeros(2, 2, 2), (:i, :j, :k))
        @test labels(reindex(tensor, :i => :u, :j => :v, :k => :w)) == (:u, :v, :w)
        @test parent(reindex(tensor, :i => :u, :j => :v, :k => :w)) === parent(tensor)

        @test labels(reindex(tensor, :a => :u, :b => :v, :c => :w)) == (:i, :j, :k)
        @test parent(reindex(tensor, :a => :u, :b => :v, :c => :w)) === parent(tensor)
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

        @test selectdim(tensor, :i, 1) == selectdim(data, 1, 1)
        @test selectdim(tensor, :j, 2) == selectdim(data, 2, 2)
    end

    @testset "view" begin
        data = rand(2, 2, 2)
        tensor = Tensor(data, (:i, :j, :k))

        @test view(tensor, 2, :, :) == view(data, 2, :, :)
        @test view(tensor, :i => 1) == view(data, 1, :, :)
        @test view(tensor, :j => 2) == view(data, :, 2, :)
        @test view(tensor, :i => 2, :k => 1) == view(data, 2, :, 1)
    end
end
@testset "Index" begin
    using Tenet: Index, links, link!, unlink!, tags, tag!, untag!, checklinks, checksize

    @test_throws DomainError Index(:_, 0)
    @test_throws DomainError Index(:_, -1)

    let tensor = Tensor(zeros(2, 2), (:i, :j)), index = Index(:i, 2)
        @test nameof(index) == :i
        @test size(index) == 2

        link!(index, tensor)
        @test tensor ∈ links(index)
        @test tensor ∉ links(copy(index))

        @test Tenet.isopenind(index)

        unlink!(index, tensor)
        @test tensor ∉ links(index)
    end

    @test_skip begin
        tensor = Tensor(zeros(2, 2), (:i, :j))
        index = Index(:i, 2)
        link!(index, tensor)

        checklinks(index)
        checksize(index)
    end

    @test_throws DimensionMismatch begin
        tensor = Tensor(zeros(2, 2), (:i, :j))
        index = Index(:i, 3)
        link!(index, tensor)
    end
    @test_throws BoundsError begin
        tensor = Tensor(zeros(2, 2), (:i, :j))
        index = Index(:_, 2)
        link!(index, tensor)
    end

    for (i, name) in enumerate((:i, :j))
        tensor = Tensor(zeros(2, 2), (:i, :j))
        index = Index(name, 2)

        @test Tenet.dim(tensor, index) == i
    end

    let index = Index(:_, 2, site = 0, tags = Set{String}(["TAG_A", "TAB_B"]))
        @test Tenet.site(index) == 0
        @test Tenet.isphysical(index)
        @test !Tenet.isvirtual(index)

        @test issetequal(tags(index), ["TAG_A", "TAB_B"])

        tag!(index, "TAG_C")
        @test "TAG_C" ∈ tags(index)

        untag!(index, "TAG_C")
        @test "TAG_C" ∉ tags(index)

        @test untag!(index, "TAG_UNEXISTANT") == tags(index)
    end

    @test begin
        index = Index(:i, 2)
        for _ in 1:3
            tensor = Tensor(zeros(2, 2), (:i, :j))
            link!(index, tensor)
        end

        Tenet.ishyperind(index)
    end
end
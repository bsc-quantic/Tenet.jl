@testset "Index" begin
    using Tenet: Index, links, link!, unlink!, tags, tag!, untag!, hastag, checklinks, checksize

    @test_throws DomainError Index(:_, 0)
    @test_throws DomainError Index(:_, -1)

    @testset "links" begin
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

        @test begin
            index = Index(:i, 2)
            for _ in 1:3
                tensor = Tensor(zeros(2, 2), (:i, :j))
                link!(index, tensor)
            end

            Tenet.ishyperind(index)
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
    end

    @testset "dim" begin
        for (i, name) in enumerate((:i, :j))
            tensor = Tensor(zeros(2, 2), (:i, :j))
            index = Index(name, 2)

            @test Tenet.dim(tensor, index) == i
        end
    end

    let index = Index(:_, 2, site = 0, tags = Set{String}(["TAG_A", "TAB_B"]))
        @test Tenet.site(index) == 0
        @test Tenet.isphysical(index)
        @test !Tenet.isvirtual(index)
    end

    @testset "tags" begin
        index = Index(:_, 2, site = 0, tags = Set{String}(["TAG_A", "TAG_B"]))

        @test issetequal(tags(index), ["TAG_A", "TAG_B"])

        tag!(index, "TAG_C")
        @test hastag(index, "TAG_C")

        untag!(index, "TAG_C")
        @test !hastag(index, "TAG_C")

        @test untag!(index, "TAG_UNEXISTANT") == tags(index)
    end

    @testset "Base.replace" begin
        index = Index(:i, 2, site = 0, tags = Set{String}(["TEST"]))
        tensor = Tensor(zeros(2, 2), (:i, :j))
        link!(index, tensor)

        new_index = replace(index, :k)
        @test nameof(new_index) == :k
        @test size(new_index) == 2
        @test hastag(new_index, "TEST")
        @test Tenet.site(new_index) == 0
        @test isempty(new_index.links)
    end
end

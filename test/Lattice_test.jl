using Tenet
using Tenet: Lattice, LatticeEdge
using Graphs
using BijectiveDicts: BijectiveIdDict

@testset "Lattice" begin
    @testset let graph = SimpleGraph(), mapping = BijectiveIdDict{Site,Int}(), lattice = Lattice(mapping, graph)
        @test lattice == zero(Lattice)
        @test nv(lattice) == 0
        @test ne(lattice) == 0
        @test isempty(vertices(lattice))
        @test isempty(edges(lattice))
        @test !has_vertex(lattice, site"1")
        @test !has_edge(lattice, site"1", site"2")
        @test !has_edge(lattice, LatticeEdge(site"1", site"2"))
        @test_throws ArgumentError neighbors(lattice, site"1")
    end

    @testset let n = 5,
        graph = path_graph(n),
        mapping = BijectiveIdDict{Site,Int}(Pair{Site,Int}[Site(i) => i for i in 1:n]),
        lattice = Lattice(mapping, graph)

        @test nv(lattice) == n
        @test ne(lattice) == n - 1

        @test issetequal(vertices(lattice), map(i -> Site(i), 1:n))
        @test issetequal(
            edges(lattice), LatticeEdge[site"1" => site"2", site"2" => site"3", site"3" => site"4", site"4" => site"5"]
        )

        for i in 1:n
            @test has_vertex(lattice, Site(i))
            @test neighbors(lattice, Site(i)) == if i == 1
                [site"2"]
            elseif i == n
                [site"4"]
            else
                [Site(i - 1), Site(i + 1)]
            end
        end

        for i in 1:(n - 1)
            @test has_edge(lattice, Site(i), Site(i + 1))
            @test has_edge(lattice, LatticeEdge(Site(i), Site(i + 1)))
        end
    end

    @testset let m = 3,
        n = 2,
        graph = grid((m, n)),
        mapping = BijectiveIdDict{Site,Int}(
            vec(Pair{Site,Int}[Site(i, j) => k for (k, (i, j)) in enumerate(Iterators.product(1:m, 1:n))])
        ),
        lattice = Lattice(mapping, graph)

        @test nv(lattice) == m * n
        @test ne(lattice) == (m - 1) * n + m * (n - 1)

        @test issetequal(vertices(lattice), map(Site, Iterators.product(1:m, 1:n)))
        @test issetequal(
            edges(lattice),
            LatticeEdge[
                site"1,1" => site"2,1",
                site"1,1" => site"1,2",
                site"2,1" => site"3,1",
                site"2,1" => site"2,2",
                site"3,1" => site"3,2",
                site"1,2" => site"2,2",
                site"2,2" => site"3,2",
            ],
        )

        for i in 1:m, j in 1:n
            site = Site(i, j)
            @test has_vertex(lattice, site)
            @test issetequal(
                neighbors(lattice, site),
                filter(
                    site -> has_vertex(lattice, site), [Site(i - 1, j), Site(i + 1, j), Site(i, j - 1), Site(i, j + 1)]
                ),
            )
        end

        for i in 1:(m - 1), j in 1:n
            @test has_edge(lattice, Site(i, j), Site(i + 1, j))
            @test has_edge(lattice, LatticeEdge(Site(i, j), Site(i + 1, j)))
        end
    end
end

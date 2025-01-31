using Tenet
using Tenet: Lattice, Bond
using Graphs

@testset "Lattice" begin
    @testset let graph = SimpleGraph(), lattice = Lattice()
        @test lattice == zero(Lattice)
        @test lattice == copy(lattice) && lattice !== copy(lattice)
        @test nv(lattice) == 0
        @test ne(lattice) == 0
        @test isempty(vertices(lattice))
        @test isempty(edges(lattice))
        @test !has_vertex(lattice, lane"1")
        @test !has_edge(lattice, lane"1", lane"2")
        @test !has_edge(lattice, Bond(lane"1", lane"2"))
        @test_throws ArgumentError neighbors(lattice, lane"1")

        add_vertex!(lattice, lane"1")
        @test nv(lattice) == 1
        @test ne(lattice) == 0
        @test issetequal(vertices(lattice), [lane"1"])
        @test isempty(edges(lattice))
        @test has_vertex(lattice, lane"1")

        add_vertex!(lattice, lane"2")
        add_edge!(lattice, lane"1", lane"2")
        @test nv(lattice) == 2
        @test ne(lattice) == 1
        @test issetequal(vertices(lattice), [lane"1", lane"2"])
        @test issetequal(edges(lattice), [Bond(lane"1", lane"2")])
        @test has_edge(lattice, lane"1", lane"2")
    end

    @testset "Chain" begin
        n = 5
        lattice = Lattice(Val(:chain), n)

        @test nv(lattice) == n
        @test ne(lattice) == n - 1

        @test issetequal(vertices(lattice), map(i -> Lane(i), 1:n))
        @test issetequal(
            edges(lattice), Bond[lane"1" => lane"2", lane"2" => lane"3", lane"3" => lane"4", lane"4" => lane"5"]
        )

        for i in 1:n
            @test has_vertex(lattice, Lane(i))
            @test neighbors(lattice, Lane(i)) == if i == 1
                [lane"2"]
            elseif i == n
                [lane"4"]
            else
                [Lane(i - 1), Lane(i + 1)]
            end
        end

        for i in 1:(n - 1)
            @test has_edge(lattice, Lane(i), Lane(i + 1))
            @test has_edge(lattice, Bond(Lane(i), Lane(i + 1)))
        end
    end

    @testset "Rectangular Grid" begin
        m = 3
        n = 2
        lattice = Lattice(Val(:rectangular), m, n)

        @test nv(lattice) == m * n
        @test ne(lattice) == (m - 1) * n + m * (n - 1)

        @test issetequal(vertices(lattice), map(Lane, Iterators.product(1:m, 1:n)))
        @test issetequal(
            edges(lattice),
            Bond[
                lane"1,1" => lane"2,1",
                lane"1,1" => lane"1,2",
                lane"2,1" => lane"3,1",
                lane"2,1" => lane"2,2",
                lane"3,1" => lane"3,2",
                lane"1,2" => lane"2,2",
                lane"2,2" => lane"3,2",
            ],
        )

        for i in 1:m, j in 1:n
            site = Lane(i, j)
            @test has_vertex(lattice, site)
            @test issetequal(
                neighbors(lattice, site),
                filter(
                    site -> has_vertex(lattice, site), [Lane(i - 1, j), Lane(i + 1, j), Lane(i, j - 1), Lane(i, j + 1)]
                ),
            )
        end

        for i in 1:(m - 1), j in 1:n
            @test has_edge(lattice, Lane(i, j), Lane(i + 1, j))
            @test has_edge(lattice, Bond(Lane(i, j), Lane(i + 1, j)))
        end
    end

    @testset "Lieb" begin
        m = 2
        n = 2
        lattice = Lattice(Val(:lieb), m, n)

        @test nv(lattice) == 21
        @test ne(lattice) == 24

        @test issetequal(
            vertices(lattice),
            map(Lane, [Lane(row, col) for row in 1:(2m + 1) for col in 1:(2n + 1) if !(row % 2 == 0 && col % 2 == 0)]),
        )
        @test issetequal(
            edges(lattice),
            Bond[
                lane"1,1" => lane"1,2",
                lane"1,1" => lane"2,1",
                lane"1,2" => lane"1,3",
                lane"1,3" => lane"1,4",
                lane"1,3" => lane"2,3",
                lane"1,4" => lane"1,5",
                lane"1,5" => lane"2,5",
                lane"2,1" => lane"3,1",
                lane"2,3" => lane"3,3",
                lane"2,5" => lane"3,5",
                lane"3,1" => lane"3,2",
                lane"3,1" => lane"4,1",
                lane"3,2" => lane"3,3",
                lane"3,3" => lane"3,4",
                lane"3,3" => lane"4,3",
                lane"3,4" => lane"3,5",
                lane"3,5" => lane"4,5",
                lane"4,1" => lane"5,1",
                lane"4,3" => lane"5,3",
                lane"4,5" => lane"5,5",
                lane"5,1" => lane"5,2",
                lane"5,2" => lane"5,3",
                lane"5,3" => lane"5,4",
                lane"5,4" => lane"5,5",
            ],
        )
    end
end

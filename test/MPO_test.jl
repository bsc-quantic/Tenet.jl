@testset "MPO" begin
    H = MPO([rand(2, 2, 4), rand(2, 2, 4, 4), rand(2, 2, 4)])
    @test socket(H) == Operator()
    @test nsites(H; set=:inputs) == 3
    @test nsites(H; set=:outputs) == 3
    @test issetequal(sites(H), [site"1", site"2", site"3", site"1'", site"2'", site"3'"])
    @test boundary(H) == Open()
    @test inds(H; at=site"1", dir=:left) == inds(H; at=site"3", dir=:right) == nothing

    arrays = [rand(2, 4, 1), rand(2, 4, 1, 3), rand(2, 4, 3)] # Default order (:o :i, :l, :r)
    H = MPO(arrays)

    @test size(tensors(H; at=Site(1))) == (2, 4, 1)
    @test size(tensors(H; at=Site(2))) == (2, 4, 1, 3)
    @test size(tensors(H; at=Site(3))) == (2, 4, 3)

    @test inds(H; at=Site(1), dir=:left) == inds(H; at=Site(3), dir=:right) === nothing
    @test inds(H; at=Site(2), dir=:left) == inds(H; at=Site(1), dir=:right) !== nothing
    @test inds(H; at=Site(3), dir=:left) == inds(H; at=Site(2), dir=:right) !== nothing

    for i in 1:length(arrays)
        @test size(H, inds(H; at=Site(i))) == 2
        @test size(H, inds(H; at=Site(i; dual=true))) == 4
    end

    arrays = [
        permutedims(arrays[1], (3, 1, 2)), permutedims(arrays[2], (4, 1, 3, 2)), permutedims(arrays[3], (1, 3, 2))
    ] # now we have (:r, :o, :l, :i)
    H = MPO(arrays; order=[:r, :o, :l, :i])

    @test size(tensors(H; at=Site(1))) == (1, 2, 4)
    @test size(tensors(H; at=Site(2))) == (3, 2, 1, 4)
    @test size(tensors(H; at=Site(3))) == (2, 3, 4)

    @test inds(H; at=Site(1), dir=:left) == inds(H; at=Site(3), dir=:right) === nothing
    @test inds(H; at=Site(2), dir=:left) == inds(H; at=Site(1), dir=:right) !== nothing
    @test inds(H; at=Site(3), dir=:left) == inds(H; at=Site(2), dir=:right) !== nothing

    for i in 1:length(arrays)
        @test size(H, inds(H; at=Site(i))) == 2
        @test size(H, inds(H; at=Site(i; dual=true))) == 4
    end

    @testset "Site" begin
        H = MPO([rand(2, 2, 2), rand(2, 2, 2, 2), rand(2, 2, 2)])

        @test isnothing(sites(H, Site(1); dir=:left))
        @test isnothing(sites(H, Site(3); dir=:right))

        @test sites(H, Site(2); dir=:left) == Site(1)
        @test sites(H, Site(3); dir=:left) == Site(2)

        @test sites(H, Site(2); dir=:right) == Site(3)
        @test sites(H, Site(1); dir=:right) == Site(2)
    end

    @testset "norm" begin
        using LinearAlgebra: norm

        n = 8
        χ = 10
        H = rand(MPO; n, maxdim=χ)

        @test socket(H) == Operator()
        @test nsites(H; set=:inputs) == n
        @test nsites(H; set=:outputs) == n
        @test issetequal(sites(H), vcat(map(Site, 1:n), map(adjoint ∘ Site, 1:n)))
        @test boundary(H) == Open()
        @test isapprox(norm(H), 1.0)
        @test maximum(last, size(H)) <= χ
    end
end

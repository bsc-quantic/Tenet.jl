@testset "MPO" begin
    H = MPO([rand(2, 2, 4), rand(2, 2, 4, 4), rand(2, 2, 4)])
    @test socket(H) == Operator()
    @test nsites(H; set=:inputs) == 3
    @test nsites(H; set=:outputs) == 3
    @test issetequal(sites(H), [site"1", site"2", site"3", site"1'", site"2'", site"3'"])
    @test boundary(H) == Open()
    @test inds(H; at=lane"1", dir=:left) == inds(H; at=lane"3", dir=:right) == nothing

    # Default order (:o :i, :l, :r)
    H = MPO([rand(2, 4, 1), rand(2, 4, 1, 3), rand(2, 4, 3)])

    @test size(tensors(H; at=site"1")) == (2, 4, 1)
    @test size(tensors(H; at=site"2")) == (2, 4, 1, 3)
    @test size(tensors(H; at=site"3")) == (2, 4, 3)

    @test inds(H; at=lane"1", dir=:left) == inds(H; at=lane"3", dir=:right) === nothing
    @test inds(H; at=lane"2", dir=:left) == inds(H; at=lane"1", dir=:right) !== nothing
    @test inds(H; at=lane"3", dir=:left) == inds(H; at=lane"2", dir=:right) !== nothing

    for i in 1:Tenet.ntensors(H)
        @test size(H, inds(H; at=Site(i))) == 2
        @test size(H, inds(H; at=Site(i; dual=true))) == 4
    end

    # now we have (:r, :o, :l, :i)
    H = MPO(
        [
            permutedims(arrays(H)[1], (3, 1, 2)),
            permutedims(arrays(H)[2], (4, 1, 3, 2)),
            permutedims(arrays(H)[3], (1, 3, 2)),
        ];
        order=[:r, :o, :l, :i],
    )

    @test size(tensors(H; at=site"1")) == (1, 2, 4)
    @test size(tensors(H; at=site"2")) == (3, 2, 1, 4)
    @test size(tensors(H; at=site"3")) == (2, 3, 4)

    @test inds(H; at=lane"1", dir=:left) == inds(H; at=lane"3", dir=:right) === nothing
    @test inds(H; at=lane"2", dir=:left) == inds(H; at=lane"1", dir=:right) !== nothing
    @test inds(H; at=lane"3", dir=:left) == inds(H; at=lane"2", dir=:right) !== nothing

    for i in 1:Tenet.ntensors(H)
        @test size(H, inds(H; at=Site(i))) == 2
        @test size(H, inds(H; at=Site(i; dual=true))) == 4
    end

    @testset "Site" begin
        H = MPO([rand(2, 2, 2), rand(2, 2, 2, 2), rand(2, 2, 2)])

        @test isnothing(sites(H, site"1"; dir=:left))
        @test isnothing(sites(H, site"3"; dir=:right))

        @test sites(H, site"2"; dir=:left) == site"1"
        @test sites(H, site"3"; dir=:left) == site"2"

        @test sites(H, site"2"; dir=:right) == site"3"
        @test sites(H, site"1"; dir=:right) == site"2"
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

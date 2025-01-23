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

    @testset "canonize!" begin
        using Tenet: isleftcanonical, isrightcanonical

        ψ = MPO([rand(4, 4, 4), rand(4, 4, 4, 4), rand(4, 4, 4, 4), rand(4, 4, 4, 4), rand(4, 4, 4)])
        canonized = canonize(ψ)

        @test form(canonized) isa Canonical

        @test length(tensors(canonized)) == 9 # 5 tensors + 4 singular values vectors
        @test isapprox(contract(transform(TensorNetwork(canonized), Tenet.HyperFlatten())), contract(ψ))
        @test isapprox(norm(ψ), norm(canonized))

        # Extract the singular values between each adjacent pair of sites in the canonized chain
        Λ = [tensors(canonized; between=(Lane(i), Lane(i + 1))) for i in 1:4]
        @test map(λ -> sum(abs2, λ), Λ) ≈ ones(length(Λ)) * norm(canonized)^2

        for i in 1:5
            canonized = canonize(ψ)

            if i == 1
                @test isleftcanonical(canonized, Lane(i))
            elseif i == 5 # in the limits of the chain, we get the norm of the state
                normalize!(tensors(canonized; bond=(Lane(i - 1), Lane(i))))
                contract!(canonized; between=(Lane(i - 1), Lane(i)), direction=:right)
                @test isleftcanonical(canonized, Lane(i))
            else
                contract!(canonized; between=(Lane(i - 1), Lane(i)), direction=:right)
                @test isleftcanonical(canonized, Lane(i))
            end
        end

        for i in 1:5
            canonized = canonize(ψ)

            if i == 1 # in the limits of the chain, we get the norm of the state
                normalize!(tensors(canonized; bond=(Lane(i), Lane(i + 1))))
                contract!(canonized; between=(Lane(i), Lane(i + 1)), direction=:left)
                @test isrightcanonical(canonized, Lane(i))
            elseif i == 5
                @test isrightcanonical(canonized, Lane(i))
            else
                contract!(canonized; between=(Lane(i), Lane(i + 1)), direction=:left)
                @test isrightcanonical(canonized, Lane(i))
            end
        end
    end

    @testset "mixed_canonize!" begin
        ψ = MPO([rand(4, 4, 4), rand(4, 4, 4, 4), rand(4, 4, 4, 4), rand(4, 4, 4, 4), rand(4, 4, 4)])

        @testset "single Site" begin
            canonized = mixed_canonize(ψ, lane"3")
            @test Tenet.check_form(canonized)

            @test form(canonized) isa MixedCanonical
            @test form(canonized).orthog_center == lane"3"

            @test isisometry(canonized, lane"1"; dir=:right)
            @test isisometry(canonized, lane"2"; dir=:right)
            @test isisometry(canonized, lane"4"; dir=:left)
            @test isisometry(canonized, lane"5"; dir=:left)

            @test contract(canonized) ≈ contract(ψ)
        end

        @testset "multiple Sites" begin
            canonized = mixed_canonize(ψ, [lane"2", lane"3"])

            @test Tenet.check_form(canonized)
            @test form(canonized) isa MixedCanonical
            @test form(canonized).orthog_center == [lane"2", lane"3"]

            @test isisometry(canonized, lane"1"; dir=:right)
            @test isisometry(canonized, lane"4"; dir=:left)
            @test isisometry(canonized, lane"5"; dir=:left)

            @test contract(canonized) ≈ contract(ψ)
        end
    end
end

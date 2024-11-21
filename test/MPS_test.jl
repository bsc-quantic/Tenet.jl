using Tenet: canonize_site, canonize_site!
using LinearAlgebra

@testset "MPS" begin
    ψ = MPS([rand(2, 2), rand(2, 2, 2), rand(2, 2)])
    @test socket(ψ) == State()
    @test nsites(ψ; set=:inputs) == 0
    @test nsites(ψ; set=:outputs) == 3
    @test issetequal(sites(ψ), [site"1", site"2", site"3"])
    @test boundary(ψ) == Open()
    @test inds(ψ; at=site"1", dir=:left) == inds(ψ; at=site"3", dir=:right) == nothing

    arrays = [rand(2, 1), rand(2, 1, 3), rand(2, 3)]
    ψ = MPS(arrays) # Default order (:o, :l, :r)
    @test size(tensors(ψ; at=site"1")) == (2, 1)
    @test size(tensors(ψ; at=site"2")) == (2, 1, 3)
    @test size(tensors(ψ; at=site"3")) == (2, 3)
    @test inds(ψ; at=site"1", dir=:left) == inds(ψ; at=site"3", dir=:right) === nothing
    @test inds(ψ; at=site"2", dir=:left) == inds(ψ; at=site"1", dir=:right)
    @test inds(ψ; at=site"3", dir=:left) == inds(ψ; at=site"2", dir=:right)

    arrays = [permutedims(arrays[1], (2, 1)), permutedims(arrays[2], (3, 1, 2)), permutedims(arrays[3], (1, 2))] # now we have (:r, :o, :l)
    ψ = MPS(arrays; order=[:r, :o, :l])
    @test size(tensors(ψ; at=site"1")) == (1, 2)
    @test size(tensors(ψ; at=site"2")) == (3, 2, 1)
    @test size(tensors(ψ; at=site"3")) == (2, 3)
    @test inds(ψ; at=site"1", dir=:left) == inds(ψ; at=site"3", dir=:right) === nothing
    @test inds(ψ; at=site"2", dir=:left) == inds(ψ; at=site"1", dir=:right) !== nothing
    @test inds(ψ; at=site"3", dir=:left) == inds(ψ; at=site"2", dir=:right) !== nothing
    @test all(i -> size(ψ, inds(ψ; at=Site(i))) == 2, 1:nsites(ψ))

    @testset "identity constructor" begin
        nsites_cases = [6, 7, 6, 7]
        physdim_cases = [3, 2, 3, 2]
        maxdim_cases = [nothing, nothing, 9, 4] # nothing means default
        expected_tensorsizes_cases = [
            [(3, 3), (3, 3, 9), (3, 9, 27), (3, 27, 9), (3, 9, 3), (3, 3)],
            [(2, 2), (2, 2, 4), (2, 4, 8), (2, 8, 8), (2, 8, 4), (2, 4, 2), (2, 2)],
            [(3, 3), (3, 3, 9), (3, 9, 9), (3, 9, 9), (3, 9, 3), (3, 3)],
            [(2, 2), (2, 2, 4), (2, 4, 4), (2, 4, 4), (2, 4, 4), (2, 4, 2), (2, 2)],
        ]

        for (nsites, physdim, expected_tensorsizes, maxdim) in
            zip(nsites_cases, physdim_cases, expected_tensorsizes_cases, maxdim_cases)
            ψ = if isnothing(maxdim)
                MPS(identity, nsites; physdim=physdim)
            else
                MPS(identity, nsites; physdim=physdim, maxdim=maxdim)
            end

            # Test the tensor dimensions
            obtained_tensorsizes = size.(tensors(ψ))
            @test obtained_tensorsizes == expected_tensorsizes

            # Test whether all tensors are the identity
            alltns = tensors(ψ)

            # - Test extreme tensors (2D) equal identity
            diagonal_2D = [fill(i, 2) for i in 1:physdim]
            @test all(delta -> alltns[1][delta...] == 1, diagonal_2D)
            @test sum(alltns[1]) == physdim
            @test all(delta -> alltns[end][delta...] == 1, diagonal_2D)
            @test sum(alltns[end]) == physdim

            # - Test bulk tensors (3D) equal identity
            diagonal_3D = [fill(i, 3) for i in 1:physdim]
            @test all(tns -> all(delta -> tns[delta...] == 1, diagonal_3D), alltns[2:(end - 1)])
            @test all(tns -> sum(tns) == physdim, alltns[2:(end - 1)])

            # Test whether the contraction gives the identity
            contracted_ψ = contract(ψ)
            diagonal_nsitesD = [fill(i, nsites) for i in 1:physdim]
            @test all(delta -> contracted_ψ[delta...] == 1, diagonal_nsitesD)
            @test sum(contracted_ψ) == physdim
        end
    end

    @testset "Site" begin
        ψ = MPS([rand(2, 2), rand(2, 2, 2), rand(2, 2)])

        @test isnothing(sites(ψ, site"1"; dir=:left))
        @test isnothing(sites(ψ, site"3"; dir=:right))

        @test sites(ψ, site"2"; dir=:left) == site"1"
        @test sites(ψ, site"3"; dir=:left) == site"2"

        @test sites(ψ, site"2"; dir=:right) == site"3"
        @test sites(ψ, site"1"; dir=:right) == site"2"
    end

    @testset "adjoint" begin
        ψ = rand(MPS; n=3, maxdim=2, eltype=ComplexF64)
        @test socket(ψ') == State(; dual=true)
        @test isapprox(contract(ψ), conj(contract(ψ')))
    end

    @testset "truncate!" begin
        @testset "NonCanonical" begin
            ψ = MPS([rand(2, 2), rand(2, 2, 2), rand(2, 2)])
            canonize_site!(ψ, Site(2); direction=:right, method=:svd)

            truncated = truncate(ψ, [site"2", site"3"]; maxdim=1)
            @test size(truncated, inds(truncated; bond=[site"2", site"3"])) == 1

            singular_values = tensors(ψ; between=(site"2", site"3"))
            truncated = truncate(ψ, [site"2", site"3"]; threshold=singular_values[2] + 0.1)
            @test size(truncated, inds(truncated; bond=[site"2", site"3"])) == 1

            # If maxdim > size(spectrum), the bond dimension is not truncated
            truncated = truncate(ψ, [site"2", site"3"]; maxdim=4)
            @test size(truncated, inds(truncated; bond=[site"2", site"3"])) == 2
        end

        @testset "Canonical" begin
            ψ = rand(MPS; n=5, maxdim=16)
            canonize!(ψ)

            truncated = truncate(ψ, [site"2", site"3"]; maxdim=2, recanonize=true)
            @test size(truncated, inds(truncated; bond=[site"2", site"3"])) == 2
            @test Tenet.check_form(truncated)
        end

        @testset "MixedCanonical" begin
            ψ = rand(MPS; n=5, maxdim=16)

            truncated = truncate(ψ, [site"2", site"3"]; maxdim=3)
            @test size(truncated, inds(truncated; bond=[site"2", site"3"])) == 3
        end
    end

    @testset "norm" begin
        using LinearAlgebra: norm

        n = 8
        χ = 10
        ψ = rand(MPS; n, maxdim=χ)

        @test socket(ψ) == State()
        @test nsites(ψ; set=:inputs) == 0
        @test nsites(ψ; set=:outputs) == n
        @test issetequal(sites(ψ), map(Site, 1:n))
        @test boundary(ψ) == Open()
        @test isapprox(norm(ψ), 1.0)
        @test maximum(last, size(ψ)) <= χ
    end

    @testset "normalize!" begin
        using LinearAlgebra: normalize!

        @testset "NonCanonical" begin
            ψ = MPS([rand(4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4)])

            normalized = normalize(ψ)
            @test norm(normalized) ≈ 1.0

            normalize!(ψ, Site(3))
            @test norm(ψ) ≈ 1.0
        end

        @testset "MixedCanonical" begin
            ψ = rand(MPS; n=5, maxdim=16)

            normalized = normalize(ψ)
            @test norm(normalized) ≈ 1.0

            normalize!(ψ, Site(3))
            @test norm(ψ) ≈ 1.0
        end

        @testset "Canonical" begin
            ψ = rand(MPS; n=5, maxdim=16)
            canonize!(ψ)

            normalized = normalize(ψ)
            @test norm(normalized) ≈ 1.0

            normalize!(ψ, Site(3))
            @test norm(ψ) ≈ 1.0
        end
    end

    @testset "canonize_site!" begin
        ψ = MPS([rand(4, 4), rand(4, 4, 4), rand(4, 4)])

        @test_throws ArgumentError canonize_site!(ψ, Site(1); direction=:left)
        @test_throws ArgumentError canonize_site!(ψ, Site(3); direction=:right)

        for method in [:qr, :svd]
            canonized = canonize_site(ψ, site"1"; direction=:right, method=method)
            @test isleftcanonical(canonized, site"1")
            @test isapprox(contract(transform(TensorNetwork(canonized), Tenet.HyperFlatten())), contract(ψ))

            canonized = canonize_site(ψ, site"2"; direction=:right, method=method)
            @test isleftcanonical(canonized, site"2")
            @test isapprox(contract(transform(TensorNetwork(canonized), Tenet.HyperFlatten())), contract(ψ))

            canonized = canonize_site(ψ, site"2"; direction=:left, method=method)
            @test isrightcanonical(canonized, site"2")
            @test isapprox(contract(transform(TensorNetwork(canonized), Tenet.HyperFlatten())), contract(ψ))

            canonized = canonize_site(ψ, site"3"; direction=:left, method=method)
            @test isrightcanonical(canonized, site"3")
            @test isapprox(contract(transform(TensorNetwork(canonized), Tenet.HyperFlatten())), contract(ψ))
        end

        # Ensure that svd creates a new tensor
        @test length(tensors(canonize_site(ψ, Site(2); direction=:left, method=:svd))) == 4
    end

    @testset "canonize!" begin
        using Tenet: isleftcanonical, isrightcanonical

        ψ = MPS([rand(4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4)])
        canonized = canonize(ψ)

        @test form(canonized) isa Canonical

        @test length(tensors(canonized)) == 9 # 5 tensors + 4 singular values vectors
        @test isapprox(contract(transform(TensorNetwork(canonized), Tenet.HyperFlatten())), contract(ψ))
        @test isapprox(norm(ψ), norm(canonized))

        # Extract the singular values between each adjacent pair of sites in the canonized chain
        Λ = [tensors(canonized; between=(Site(i), Site(i + 1))) for i in 1:4]
        @test map(λ -> sum(abs2, λ), Λ) ≈ ones(length(Λ)) * norm(canonized)^2

        for i in 1:5
            canonized = canonize(ψ)

            if i == 1
                @test isleftcanonical(canonized, Site(i))
            elseif i == 5 # in the limits of the chain, we get the norm of the state
                normalize!(tensors(canonized; bond=(Site(i - 1), Site(i))))
                contract!(canonized; between=(Site(i - 1), Site(i)), direction=:right)
                @test isleftcanonical(canonized, Site(i))
            else
                contract!(canonized; between=(Site(i - 1), Site(i)), direction=:right)
                @test isleftcanonical(canonized, Site(i))
            end
        end

        for i in 1:5
            canonized = canonize(ψ)

            if i == 1 # in the limits of the chain, we get the norm of the state
                normalize!(tensors(canonized; bond=(Site(i), Site(i + 1))))
                contract!(canonized; between=(Site(i), Site(i + 1)), direction=:left)
                @test isrightcanonical(canonized, Site(i))
            elseif i == 5
                @test isrightcanonical(canonized, Site(i))
            else
                contract!(canonized; between=(Site(i), Site(i + 1)), direction=:left)
                @test isrightcanonical(canonized, Site(i))
            end
        end
    end

    @testset "mixed_canonize!" begin
        @testset "single Site" begin
            ψ = MPS([rand(4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4)])
            canonized = mixed_canonize(ψ, site"3")
            @test Tenet.check_form(canonized)

            @test form(canonized) isa MixedCanonical
            @test form(canonized).orthog_center == site"3"

            @test isisometry(canonized, site"1"; dir=:right)
            @test isisometry(canonized, site"2"; dir=:right)
            @test isisometry(canonized, site"4"; dir=:left)
            @test isisometry(canonized, site"5"; dir=:left)

            @test contract(canonized) ≈ contract(ψ)
        end

        @testset "multiple Sites" begin
            ψ = MPS([rand(4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4)])
            canonized = mixed_canonize(ψ, [site"2", site"3"])

            @test Tenet.check_form(canonized)
            @test form(canonized) isa MixedCanonical
            @test form(canonized).orthog_center == [site"2", site"3"]

            @test isisometry(canonized, site"1"; dir=:right)
            @test isisometry(canonized, site"4"; dir=:left)
            @test isisometry(canonized, site"5"; dir=:left)

            @test contract(canonized) ≈ contract(ψ)
        end
    end

    @testset "expect" begin
        i, j = 2, 3
        mat = reshape(kron(LinearAlgebra.I(2), LinearAlgebra.I(2)), 2, 2, 2, 2)
        gate = Quantum(mat, [Site(i), Site(j), Site(i; dual=true), Site(j; dual=true)])
        ψ = MPS([rand(2, 2), rand(2, 2, 2), rand(2, 2, 2), rand(2, 2, 2), rand(2, 2)])

        @test isapprox(expect(ψ, [gate]), norm(ψ)^2)
    end

    @testset "evolve!" begin
        @testset "one site" begin
            i = 2
            mat = reshape(LinearAlgebra.I(2), 2, 2)
            gate = Quantum(mat, [Site(i), Site(i; dual=true)])
            ψ = MPS([rand(2, 2), rand(2, 2, 2), rand(2, 2, 2), rand(2, 2, 2), rand(2, 2)])

            @testset "NonCanonical" begin
                ϕ = deepcopy(ψ)
                evolve!(ϕ, gate; threshold=1e-14)
                @test length(tensors(ϕ)) == 5
                @test issetequal(size.(tensors(ϕ)), [(2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2)])
                @test isapprox(contract(ϕ), contract(ψ))
            end

            @testset "Canonical" begin
                ϕ = deepcopy(ψ)
                canonize!(ϕ)
                evolve!(ϕ, gate; threshold=1e-14)
                @test issetequal(size.(tensors(ϕ)), [(2, 2), (2,), (2, 2, 2), (2,), (2, 2, 2), (2,), (2, 2)])
                @test isapprox(contract(ϕ), contract(ψ))
            end
        end

        @testset "two sites" begin
            mat = reshape(LinearAlgebra.I(4), 2, 2, 2, 2)
            gate = Quantum(mat, [site"2", site"3", site"2'", site"3'"])
            ψ = MPS([rand(2, 2), rand(2, 2, 2), rand(2, 2, 2), rand(2, 2)])

            @testset "NonCanonical" begin
                ϕ = deepcopy(ψ)
                evolve!(ϕ, gate; threshold=1e-14)
                @test length(tensors(ϕ)) == 5
                @test issetequal(size.(tensors(ϕ)), [(2, 2), (2, 2, 2), (2,), (2, 2, 2), (2, 2, 2), (2, 2)])
                @test isapprox(contract(ϕ), contract(ψ))
            end

            @testset "Canonical" begin
                ψ = rand(MPS; n=5, maxdim=20)
                ϕ = deepcopy(ψ)
                canonize!(ψ)
                evolved = evolve!(deepcopy(ψ), gate)

                @test Tenet.check_form(evolved)
                @test isapprox(contract(evolved), contract(ϕ)) # Identity gate should not change the state

                # Ensure that the original MixedCanonical state evolves into the same state as the canonicalized one
                @test contract(evolve!(ϕ, gate; threshold=1e-14)) ≈ contract(ψ)
            end
        end
    end

    # TODO rename when method is renamed
    @testset "contract between" begin
        ψ = rand(MPS; n=5, maxdim=20)
        let canonized = canonize(ψ)
            @test_throws ArgumentError contract!(canonized; between=(site"1", site"2"), direction=:dummy)
        end

        canonized = canonize(ψ)

        for i in 1:4
            contract_some = contract(canonized; between=(Site(i), Site(i + 1)))
            Bᵢ = tensors(contract_some; at=Site(i))

            @test isapprox(contract(contract_some), contract(ψ))
            @test_throws Tenet.MissingSchmidtCoefficientsException tensors(
                contract_some; between=(Site(i), Site(i + 1))
            )

            @test isrightcanonical(contract_some, Site(i))
            @test isleftcanonical(contract(canonized; between=(Site(i), Site(i + 1)), direction=:right), Site(i + 1))

            Γᵢ = tensors(canonized; at=Site(i))
            Λᵢ₊₁ = tensors(canonized; between=(Site(i), Site(i + 1)))
            @test Bᵢ ≈ contract(Γᵢ, Λᵢ₊₁; dims=())
        end
    end
end

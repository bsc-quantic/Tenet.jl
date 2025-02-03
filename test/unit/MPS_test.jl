using Tenet: nsites, State, canonize_site, canonize_site!
using LinearAlgebra

@testset "Interfaces" begin
    @testset "case 1" begin
        ψ = MPS([rand(2, 2), rand(2, 2, 2), rand(2, 2)])

        test_tensornetwork(ψ; contract_mut=false)
        test_pluggable(ψ)
        test_ansatz(ψ)

        @test socket(ψ) == State()
        @test nsites(ψ; set=:inputs) == 0
        @test nsites(ψ; set=:outputs) == 3
        @test issetequal(sites(ψ), [site"1", site"2", site"3"])
        @test boundary(ψ) == Open()
        @test inds(ψ; at=site"1", dir=:left) == inds(ψ; at=site"3", dir=:right) == nothing
    end

    @testset "case 2" begin
        ψ = MPS([rand(2, 1), rand(2, 1, 3), rand(2, 3)]) # Default order (:o, :l, :r)

        test_tensornetwork(ψ; contract_mut=false)
        test_pluggable(ψ)
        test_ansatz(ψ)

        @test size(tensors(ψ; at=site"1")) == (2, 1)
        @test size(tensors(ψ; at=site"2")) == (2, 1, 3)
        @test size(tensors(ψ; at=site"3")) == (2, 3)
        @test inds(ψ; at=lane"1", dir=:left) == inds(ψ; at=lane"3", dir=:right) === nothing
        @test inds(ψ; at=lane"2", dir=:left) == inds(ψ; at=lane"1", dir=:right)
        @test inds(ψ; at=lane"3", dir=:left) == inds(ψ; at=lane"2", dir=:right)
    end

    @testset "case 3: order = [:r, :o, :l]" begin
        ψ = MPS([rand(1, 2), rand(3, 2, 1), rand(2, 3)]; order=[:r, :o, :l])

        test_tensornetwork(ψ; contract_mut=false)
        test_pluggable(ψ)
        test_ansatz(ψ)

        @test size(tensors(ψ; at=site"1")) == (1, 2)
        @test size(tensors(ψ; at=site"2")) == (3, 2, 1)
        @test size(tensors(ψ; at=site"3")) == (2, 3)
        @test inds(ψ; at=lane"1", dir=:left) == inds(ψ; at=lane"3", dir=:right) === nothing
        @test inds(ψ; at=lane"2", dir=:left) == inds(ψ; at=lane"1", dir=:right) !== nothing
        @test inds(ψ; at=lane"3", dir=:left) == inds(ψ; at=lane"2", dir=:right) !== nothing
        @test all(i -> size(ψ, inds(ψ; at=Site(i))) == 2, 1:nsites(ψ))
    end
end

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

@testset "sites" begin
    ψ = MPS([rand(2, 2), rand(2, 2, 2), rand(2, 2)])

    @test isnothing(sites(ψ, site"1"; dir=:left))
    @test isnothing(sites(ψ, site"3"; dir=:right))

    @test sites(ψ, site"2"; dir=:left) == site"1"
    @test sites(ψ, site"3"; dir=:left) == site"2"

    @test sites(ψ, site"2"; dir=:right) == site"3"
    @test sites(ψ, site"1"; dir=:right) == site"2"
end

@testset "lanes" begin
    ψ = MPS([rand(2, 2), rand(2, 2, 2), rand(2, 2)])

    @test isnothing(lanes(ψ, lane"1"; dir=:left))
    @test isnothing(lanes(ψ, lane"3"; dir=:right))

    @test lanes(ψ, lane"2"; dir=:left) == lane"1"
    @test lanes(ψ, lane"3"; dir=:left) == lane"2"

    @test lanes(ψ, lane"2"; dir=:right) == lane"3"
    @test lanes(ψ, lane"1"; dir=:right) == lane"2"
end

@testset "adjoint" begin
    ψ = rand(MPS; n=3, maxdim=2, eltype=ComplexF64)
    @test socket(ψ') == State(; dual=true)
    @test isapprox(contract(ψ), conj(contract(ψ')))
end

@testset "truncate!" begin
    @testset "NonCanonical" begin
        ψ = MPS([rand(2, 2), rand(2, 2, 2), rand(2, 2)])
        canonize_site!(ψ, lane"2"; dir=:right, method=:svd)

        truncated = truncate(ψ, [lane"2", lane"3"]; maxdim=1)
        @test size(truncated, inds(truncated; bond=[lane"2", lane"3"])) == 1

        singular_values = tensors(ψ; bond=(lane"2", lane"3"))
        truncated = truncate(ψ, [lane"2", lane"3"]; threshold=singular_values[2] + 0.1)
        @test size(truncated, inds(truncated; bond=[lane"2", lane"3"])) == 1

        # If maxdim > size(spectrum), the bond dimension is not truncated
        truncated = truncate(ψ, [lane"2", lane"3"]; maxdim=4)
        @test size(truncated, inds(truncated; bond=[lane"2", lane"3"])) == 2

        normalize!(ψ)
        truncated = truncate(ψ, [lane"2", lane"3"]; maxdim=1, normalize=true)
        @test norm(truncated) ≈ 1.0
    end

    @testset "MixedCanonical" begin
        ψ = rand(MPS; n=5, maxdim=16)

        truncated = truncate(ψ, [lane"2", lane"3"]; maxdim=3)
        @test size(truncated, inds(truncated; bond=[lane"2", lane"3"])) == 3

        truncated = truncate(ψ, [lane"2", lane"3"]; maxdim=3, normalize=true)
        @test norm(truncated) ≈ 1.0
    end

    @testset "Canonical" begin
        ψ = rand(MPS; n=5, maxdim=16)
        canonize!(ψ)

        truncated = truncate(ψ, [lane"2", lane"3"]; maxdim=2, canonize=true, normalize=true)
        @test size(truncated, inds(truncated; bond=[lane"2", lane"3"])) == 2
        @test Tenet.check_form(truncated)
        @test norm(truncated) ≈ 1.0

        truncated = truncate(ψ, [lane"2", lane"3"]; maxdim=2, canonize=false, normalize=true)
        @test norm(truncated) ≈ 1.0
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
    using LinearAlgebra: normalize, normalize!

    @testset "NonCanonical" begin
        ψ = MPS([rand(4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4)])

        normalized = normalize(ψ)
        @test norm(normalized) ≈ 1.0

        normalize!(ψ, lane"3")
        @test norm(ψ) ≈ 1.0
    end

    @testset "MixedCanonical" begin
        ψ = rand(MPS; n=5, maxdim=16)

        # Perturb the state to make it non-normalized
        t = tensors(ψ; at=lane"3")
        replace!(ψ, t => Tensor(rand(size(t)...), inds(t)))

        normalized = normalize(ψ)
        @test norm(normalized) ≈ 1.0

        normalize!(ψ, lane"3")
        @test norm(ψ) ≈ 1.0
    end

    @testset "Canonical" begin
        ψ = MPS([rand(4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4)])
        canonize!(ψ)

        normalized = normalize(ψ)
        @test norm(normalized) ≈ 1.0

        normalize!(ψ, (lane"3", lane"4"))
        @test norm(ψ) ≈ 1.0
    end
end

@testset "canonize_site!" begin
    ψ = MPS([rand(4, 4), rand(4, 4, 4), rand(4, 4)])

    @test_throws ArgumentError canonize_site!(ψ, lane"1"; dir=:left)
    @test_throws ArgumentError canonize_site!(ψ, lane"3"; dir=:right)

    for method in [:qr, :svd]
        canonized = canonize_site(ψ, lane"1"; dir=:right, method=method)
        @test isisometry(canonized, lane"1"; dir=:right)
        @test isapprox(contract(transform(TensorNetwork(canonized), Tenet.HyperFlatten())), contract(ψ))

        canonized = canonize_site(ψ, lane"2"; dir=:right, method=method)
        @test isisometry(canonized, lane"2"; dir=:right)
        @test isapprox(contract(transform(TensorNetwork(canonized), Tenet.HyperFlatten())), contract(ψ))

        canonized = canonize_site(ψ, lane"2"; dir=:left, method=method)
        @test isisometry(canonized, lane"2"; dir=:left)
        @test isapprox(contract(transform(TensorNetwork(canonized), Tenet.HyperFlatten())), contract(ψ))

        canonized = canonize_site(ψ, lane"3"; dir=:left, method=method)
        @test isisometry(canonized, lane"3"; dir=:left)
        @test isapprox(contract(transform(TensorNetwork(canonized), Tenet.HyperFlatten())), contract(ψ))
    end

    # Ensure that svd creates a new tensor
    @test length(tensors(canonize_site(ψ, lane"2"; dir=:left, method=:svd))) == 4
end

@testset "canonize!" begin
    ψ = MPS([rand(4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4)])
    canonized = canonize(ψ)

    @test form(canonized) isa Canonical

    @test length(tensors(canonized)) == 9 # 5 tensors + 4 singular values vectors
    @test isapprox(contract(transform(TensorNetwork(canonized), Tenet.HyperFlatten())), contract(ψ))
    @test isapprox(norm(ψ), norm(canonized))

    # Extract the singular values between each adjacent pair of sites in the canonized chain
    Λ = [tensors(canonized; bond=(Lane(i), Lane(i + 1))) for i in 1:4]

    norm_psi = norm(ψ)
    @test all(λ -> sqrt(sum(abs2, λ)) ≈ norm_psi, Λ)

    for i in 1:5
        canonized = canonize(ψ)

        if i == 1
            @test isisometry(canonized, Lane(i); dir=:right)
        elseif i == 5 # in the limits of the chain, we get the norm of the state
            normalize!(tensors(canonized; bond=(Lane(i - 1), Lane(i))))
            contract!(canonized; bond=(Lane(i - 1), Lane(i)), dir=:right)
            @test isisometry(canonized, Lane(i); dir=:right)
        else
            contract!(canonized; bond=(Lane(i - 1), Lane(i)), dir=:right)
            @test isisometry(canonized, Lane(i); dir=:right)
        end
    end

    for i in 1:5
        canonized = canonize(ψ)

        if i == 1 # in the limits of the chain, we get the norm of the state
            normalize!(tensors(canonized; bond=(Lane(i), Lane(i + 1))))
            contract!(canonized; bond=(Lane(i), Lane(i + 1)), dir=:left)
            @test isisometry(canonized, Lane(i); dir=:left)
        elseif i == 5
            @test isisometry(canonized, Lane(i); dir=:left)
        else
            contract!(canonized; bond=(Lane(i), Lane(i + 1)), dir=:left)
            @test isisometry(canonized, Lane(i); dir=:left)
        end
    end
end

@testset "mixed_canonize!" begin
    @testset "single Site" begin
        ψ = MPS([rand(4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4)])
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
        ψ = MPS([rand(4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4)])
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

@testset "expect" begin
    i, j = 2, 3
    mat = reshape(kron(LinearAlgebra.I(2), LinearAlgebra.I(2)), 2, 2, 2, 2)
    gate = Gate(mat, [Site(i), Site(j), Site(i; dual=true), Site(j; dual=true)])
    ψ = MPS([rand(2, 2), rand(2, 2, 2), rand(2, 2, 2), rand(2, 2, 2), rand(2, 2)])

    @test expect(ψ, Quantum.([gate])) ≈ norm(ψ)^2
end

@testset "evolve!" begin
    @testset "one site" begin
        i = 2
        mat = reshape(LinearAlgebra.I(2), 2, 2)
        gate = Gate(mat, [Site(i), Site(i; dual=true)])
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
        gate = Gate(mat, [site"2", site"3", site"2'", site"3'"])

        @testset "NonCanonical" begin
            ψ = MPS([rand(2, 2), rand(2, 2, 2), rand(2, 2, 2), rand(2, 2)])
            ϕ = deepcopy(ψ)
            evolve!(ϕ, gate; threshold=1e-14)
            @test length(tensors(ϕ)) == 5
            @test issetequal(size.(tensors(ϕ)), [(2, 2), (2, 2, 2), (2,), (2, 2, 2), (2, 2, 2), (2, 2)])
            @test isapprox(contract(ϕ), contract(ψ))

            evolved = evolve!(normalize(ψ), gate; maxdim=1, normalize=true)
            @test norm(evolved) ≈ 1.0
        end

        @testset "Canonical" begin
            ψ = MPS([rand(2, 2), rand(2, 2, 2), rand(2, 2, 2), rand(2, 2, 2), rand(2, 2)])
            normalize!(ψ)
            ϕ = deepcopy(ψ)

            canonize!(ψ)

            evolved = evolve!(deepcopy(ψ), gate)
            @test Tenet.check_form(evolved)
            @test isapprox(contract(evolved), contract(ϕ)) # Identity gate should not change the state

            # Ensure that the original MixedCanonical state evolves into the same state as the canonicalized one
            @test contract(ψ) ≈ contract(evolve!(ϕ, gate; threshold=1e-14))

            evolved = evolve!(deepcopy(ψ), gate; maxdim=1, normalize=true, canonize=true)
            @test norm(evolved) ≈ 1.0
            @test Tenet.check_form(evolved)

            evolved = evolve!(deepcopy(ψ), gate; maxdim=1, normalize=true, canonize=false)
            @test norm(evolved) ≈ 1.0
            @test_throws ArgumentError Tenet.check_form(evolved)
        end
    end

    @testset "MPO evolution" begin
        ψ = MPS([rand(2, 2), rand(2, 2, 2), rand(2, 2, 2), rand(2, 2, 2), rand(2, 2)])
        normalize!(ψ)
        mpo = rand(MPO; n=5, maxdim=8)

        ϕ_1 = deepcopy(ψ)
        ϕ_2 = deepcopy(ψ)
        ϕ_3 = deepcopy(ψ)

        @testset "NonCanonical" begin
            evolve!(ϕ_1, mpo)
            @test length(tensors(ϕ_1)) == 5
            @test norm(ϕ_1) ≈ 1.0

            evolved = evolve!(deepcopy(ψ), mpo; maxdim=3)
            @test all(x -> x ≤ 3, vcat([collect(t) for t in vec(size.(tensors(evolved)))]...))
            @test norm(evolved) ≈ 1.0
        end

        @testset "Canonical" begin
            canonize!(ϕ_2)
            evolve!(ϕ_2, mpo)
            @test length(tensors(ϕ_2)) == 5 + 4
            @test form(ϕ_2) == Canonical()
            @test Tenet.check_form(ϕ_2)

            evolved = evolve!(deepcopy(canonize!(ψ)), mpo; maxdim=3)
            @test all(x -> x ≤ 3, vcat([collect(t) for t in vec(size.(tensors(evolved)))]...))
            @test form(evolved) == Canonical()
            @test Tenet.check_form(evolved)
        end

        @testset "MixedCanonical" begin
            mixed_canonize!(ϕ_3, lane"3")
            evolve!(ϕ_3, mpo)
            @test length(tensors(ϕ_3)) == 5
            @test form(ϕ_3) == MixedCanonical(lane"3")
            @test norm(ϕ_3) ≈ 1.0
            @test Tenet.check_form(ϕ_3)

            evolved = evolve!(deepcopy(mixed_canonize!(ψ, lane"3")), mpo; maxdim=3)
            @test all(x -> x ≤ 3, vcat([collect(t) for t in vec(size.(tensors(evolved)))]...))
            @test form(evolved) == MixedCanonical(lane"3")
            @test norm(evolved) ≈ 1.0
            @test Tenet.check_form(evolved)
        end

        t1 = contract(ϕ_1)
        t2 = contract(ϕ_2)
        t3 = contract(ϕ_3)

        @test t1 ≈ t2 ≈ t3
        @test only(overlap(ϕ_1, ϕ_2)) ≈ only(overlap(ϕ_1, ϕ_3)) ≈ only(overlap(ϕ_2, ϕ_3)) ≈ 1.0
    end
end

# TODO rename when method is renamed
@testset "contract bond" begin
    ψ = rand(MPS; n=5, maxdim=20)
    let canonized = canonize(ψ)
        @test_throws ArgumentError contract!(canonized; bond=(lane"1", lane"2"), dir=:dummy)
    end

    canonized = canonize(ψ)

    for i in 1:4
        contract_some = contract(canonized; bond=(Lane(i), Lane(i + 1)))
        Bᵢ = tensors(contract_some; at=Lane(i))

        @test isapprox(contract(contract_some), contract(ψ))
        @test_throws Tenet.MissingSchmidtCoefficientsException tensors(contract_some; bond=(Lane(i), Lane(i + 1)))

        @test isisometry(contract_some, Lane(i); dir=:left)
        @test isisometry(contract(canonized; bond=(Lane(i), Lane(i + 1)), dir=:right), Lane(i + 1); dir=:right)

        Γᵢ = tensors(canonized; at=Lane(i))
        Λᵢ₊₁ = tensors(canonized; bond=(Lane(i), Lane(i + 1)))
        @test Bᵢ ≈ contract(Γᵢ, Λᵢ₊₁; dims=())
    end
end

using Test
using Tenet
using LinearAlgebra

@testset "hasinterface" begin
    @test Tenet.hasinterface(Tenet.TensorNetworkInterface(), MPS)
    @test Tenet.hasinterface(Tenet.PluggableInterface(), MPS)
    @test Tenet.hasinterface(Tenet.AnsatzInterface(), MPS)
end

@testset "constructors" begin
    @testset "from arrays" begin
        @test MPS([zeros(2, 2), zeros(2, 2, 2), zeros(2, 2)]) isa MPS

        # default order (:o, :l, :r)
        @test MPS([zeros(2, 1), zeros(2, 1, 3), zeros(2, 3)]) isa MPS

        @test MPS([zeros(1, 2), zeros(3, 2, 1), zeros(2, 3)]; order=[:r, :o, :l]) isa MPS
    end

    @testset "rand" begin
        @test rand(MPS; n=3, maxdim=2) isa MPS
    end
end

@testset "core" begin
    ψ = MPS([rand(2, 1), rand(2, 1, 3), rand(2, 3)]) # Default order (:o, :l, :r)

    @test ntensors(ψ) == 3
    @test ninds(ψ) == 5

    @test issetequal(lanes(ψ), [lane"1", lane"2", lane"3"])
    @test issetequal(sites(ψ), [site"1", site"2", site"3"])
    @test issetequal(bonds(ψ), [Bond(lane"1", lane"2"), Bond(lane"2", lane"3")])

    @test Tenet.socket(ψ) == Tenet.State()
    @test nsites(ψ; set=:inputs) == 0
    @test nsites(ψ; set=:outputs) == 3
    @test nlanes(ψ) == 3

    @test size(tensors(ψ; at=lane"1")) == (2, 1)
    @test size(tensors(ψ; at=lane"2")) == (2, 1, 3)
    @test size(tensors(ψ; at=lane"3")) == (2, 3)

    @test inds(ψ; at=lane"1", dir=:left) == inds(ψ; at=lane"3", dir=:right) === nothing
    @test inds(ψ; at=lane"2", dir=:left) == inds(ψ; at=lane"1", dir=:right)
    @test inds(ψ; at=lane"3", dir=:left) == inds(ψ; at=lane"2", dir=:right)

    @test isnothing(lanes(ψ, lane"1"; dir=:left))
    @test isnothing(lanes(ψ, lane"3"; dir=:right))

    @test lanes(ψ, lane"2"; dir=:left) == lane"1"
    @test lanes(ψ, lane"3"; dir=:left) == lane"2"

    @test lanes(ψ, lane"2"; dir=:right) == lane"3"
    @test lanes(ψ, lane"1"; dir=:right) == lane"2"
end

@testset "adjoint" begin
    ψ = rand(MPS; n=3, maxdim=2, eltype=ComplexF64)
    @test Tenet.socket(ψ') == Tenet.State(; dual=true)
    @test isapprox(contract(ψ), conj(contract(ψ')))
end

@testset "canonize" begin
    @testset "canonize_site!" begin
        ψ = MPS([rand(4, 4), rand(4, 4, 4), rand(4, 4)])
        ψ_tensor = contract(ψ)

        @testset let ψ = copy(ψ)
            @test_throws ArgumentError Tenet.canonize_site!(ψ, lane"1"; dir=:left)
            @test_throws ArgumentError Tenet.canonize_site!(ψ, lane"3"; dir=:right)
        end

        @testset "method = qr" begin
            canonized = Tenet.canonize_site(ψ, lane"1"; dir=:right, method=:qr)
            @test isisometry(canonized, lane"1", :right)
            @test isapprox(contract(canonized), ψ_tensor)

            canonized = Tenet.canonize_site(ψ, lane"2"; dir=:right, method=:qr)
            @test isisometry(canonized, lane"2", :right)
            @test isapprox(contract(canonized), ψ_tensor)

            canonized = Tenet.canonize_site(ψ, lane"2"; dir=:left, method=:qr)
            @test isisometry(canonized, lane"2", :left)
            @test isapprox(contract(canonized), ψ_tensor)

            canonized = Tenet.canonize_site(ψ, lane"3"; dir=:left, method=:qr)
            @test isisometry(canonized, lane"3", :left)
            @test isapprox(contract(canonized), ψ_tensor)
        end

        @testset "method = svd" begin
            canonized = Tenet.canonize_site(ψ, lane"1"; dir=:right, method=:svd)
            @test isisometry(canonized, lane"1", :right)
            @test isapprox(contract(canonized), ψ_tensor)

            canonized = Tenet.canonize_site(ψ, lane"2"; dir=:right, method=:svd)
            @test isisometry(canonized, lane"2", :right)
            @test isapprox(contract(canonized), ψ_tensor)

            canonized = Tenet.canonize_site(ψ, lane"2"; dir=:left, method=:svd)
            @test isisometry(canonized, lane"2", :left)
            @test isapprox(contract(canonized), ψ_tensor)

            canonized = Tenet.canonize_site(ψ, lane"3"; dir=:left, method=:svd)
            @test isisometry(canonized, lane"3", :left)
            @test isapprox(contract(canonized), ψ_tensor)

            # do not absorb (1 tensor more)
            canonized = Tenet.canonize_site(ψ, lane"1"; dir=:right, method=:svd, absorb=nothing)
            @test isisometry(canonized, lane"1", :right)
            @test isapprox(contract(canonized), ψ_tensor)
            @test ntensors(canonized) == ntensors(ψ) + 1

            # absorb left (not isometry)
            canonized = Tenet.canonize_site(ψ, lane"1"; dir=:right, method=:svd, absorb=:left)
            @test !isisometry(canonized, lane"1", :right)
            @test isapprox(contract(canonized), ψ_tensor)

            # absorb both (not isommetry)
            canonized = Tenet.canonize_site(ψ, lane"1"; dir=:right, method=:svd, absorb=:both)
            @test !isisometry(canonized, lane"1", :right)
            @test isapprox(contract(canonized), ψ_tensor)
        end
    end

    @testset "canonize! to Canonical form" begin
        ψ = MPS([rand(4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4)])
        canonized = canonize(ψ)

        @test form(canonized) isa Canonical
        @test ntensors(canonized) == ntensors(ψ) + nbonds(ψ)
        @test isapprox(contract(canonized), contract(ψ))
        @test isapprox(norm(canonized), norm(ψ))

        # Extract the singular values between each adjacent pair of sites in the canonized chain
        Λ = [tensors(canonized; bond=(Lane(i), Lane(i + 1))) for i in 1:4]

        norm_psi = norm(ψ)
        @test all(λ -> sqrt(sum(abs2, λ)) ≈ norm_psi, Λ)

        canonized = canonize(ψ)

        # on boundaries, you have isommetries
        @test isisometry(canonized, lane"1", :right)
        @test isisometry(canonized, lane"5", :left)

        canonized_right = absorb(canonized, Bond(lane"1", lane"2"), :right)
        @test isisometry(canonized_right, lane"2", :right)

        canonized_right = absorb(canonized, Bond(lane"2", lane"3"), :right)
        @test isisometry(canonized_right, lane"3", :right)

        canonized_right = absorb(canonized, Bond(lane"3", lane"4"), :right)
        @test isisometry(canonized_right, lane"4", :right)

        canonized_left = absorb(canonized, Bond(lane"2", lane"3"), :left)
        @test isisometry(canonized_left, lane"2", :left)

        canonized_left = absorb(canonized, Bond(lane"3", lane"4"), :left)
        @test isisometry(canonized_left, lane"3", :left)

        canonized_left = absorb(canonized, Bond(lane"4", lane"5"), :left)
        @test isisometry(canonized_left, lane"4", :left)
    end

    @testset "canonize! to MixedCanonical form" begin
        @testset "single Site" begin
            ψ = MPS([rand(4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4)])
            canonized = canonize(ψ, lane"3")
            @test Tenet.checkform(canonized)

            @test form(canonized) isa MixedCanonical
            @test form(canonized).orthog_center == lane"3"

            @test isisometry(canonized, lane"1", :right)
            @test isisometry(canonized, lane"2", :right)
            @test isisometry(canonized, lane"4", :left)
            @test isisometry(canonized, lane"5", :left)

            @test contract(canonized) ≈ contract(ψ)
        end

        @testset "multiple Sites" begin
            ψ = MPS([rand(4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4)])
            canonized = canonize(ψ, [lane"2", lane"3"])

            @test Tenet.checkform(canonized)
            @test form(canonized) isa MixedCanonical
            @test form(canonized).orthog_center == [lane"2", lane"3"]

            @test isisometry(canonized, lane"1", :right)
            @test isisometry(canonized, lane"4", :left)
            @test isisometry(canonized, lane"5", :left)

            @test contract(canonized) ≈ contract(ψ)
        end
    end
end

@testset "norm" begin
    using LinearAlgebra: norm

    ψ = MPS([rand(2, 2), rand(2, 2, 2), rand(2, 2)])
    norm_value = contract(stack(ψ, ψ')) |> only |> sqrt

    @testset "NonCanonical" begin
        @test form(ψ) isa NonCanonical
        @test norm(ψ) ≈ norm_value
    end

    @testset "MixedCanonical" begin
        ϕ = canonize(ψ, MixedCanonical(lane"2"))
        @test form(ϕ) isa MixedCanonical
        @test norm(ϕ) ≈ norm_value

        # Perturb the state to make it non-normalized
        orthog_center = tensors(ϕ; at=lane"3")
        orthog_center ./= 2
        @test norm(ϕ) ≈ norm_value / 2
    end

    @testset "Canonical" begin
        ϕ = canonize(ψ)
        @test form(ϕ) isa Canonical
        @test norm(ϕ) ≈ norm_value
    end
end

@testset "normalize!" begin
    using LinearAlgebra: normalize, normalize!
    ψ = MPS([rand(4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4)])

    @testset "NonCanonical" begin
        @test norm(normalize(ψ)) ≈ 1.0
        @test norm(normalize(ψ, lane"3")) ≈ 1.0
    end

    @testset "MixedCanonical" begin
        ϕ = canonize(ψ, MixedCanonical(lane"3"))
        normalize!(ϕ)
        @test norm(ϕ) ≈ 1.0
    end

    @testset "Canonical" begin
        ϕ = canonize(ψ)
        normalize!(ϕ)
        @test norm(ϕ) ≈ 1.0

        Λ34 = tensors(ϕ; bond=(lane"3", lane"4"))
        Λ34 ./= 2
        @test norm(ϕ) ≈ 0.5

        normalize!(ψ, Bond(lane"3", lane"4"))
        @test norm(ψ) ≈ 1.0
    end
end

@testset "truncate!" begin
    @testset "NonCanonical" begin
        ψ = MPS([rand(2, 2), rand(2, 2, 2), rand(2, 2)])
        # canonize_site!(ψ, lane"2"; dir=:right; method=:svd)

        truncated = truncate(ψ, Bond(lane"2", lane"3"); maxdim=1)
        @test size(truncated, inds(truncated; bond=Bond(lane"2", lane"3"))) == 1

        # singular_values = tensors(ψ; bond=Bond(lane"2", lane"3"))
        # truncated = truncate(ψ, Bond(lane"2", lane"3"); threshold=singular_values[2] + 0.1)
        # @test size(truncated, inds(truncated; bond=[lane"2", lane"3"])) == 1

        # If maxdim > size(spectrum), the bond dimension is not truncated
        truncated = truncate(ψ, Bond(lane"2", lane"3"); maxdim=4)
        @test size(truncated, inds(truncated; bond=Bond(lane"2", lane"3"))) == 2

        # normalize!(ψ)
        # truncated = truncate(ψ, [lane"2", lane"3"]; maxdim=1, normalize=true)
        # @test norm(truncated) ≈ 1.0
    end

    @testset "MixedCanonical" begin
        ψ = rand(MPS; n=5, maxdim=16)

        truncated = truncate(ψ, Bond(lane"2", lane"3"); maxdim=3)
        @test size(truncated, inds(truncated; bond=Bond(lane"2", lane"3"))) == 3
    end

    @testset "Canonical" begin
        ψ = rand(MPS; n=5, maxdim=16)
        canonize!(ψ)

        truncated = truncate(ψ, Bond(lane"2", lane"3"); maxdim=2)
        @test size(truncated, inds(truncated; bond=Bond(lane"2", lane"3"))) == 2

        # truncation losses canonicity, so it throws an error
        @test_throws AssertionError Tenet.checkform(truncated)
    end
end

@testset "expect" begin
    i, j = 2, 3
    mat = reshape(kron(LinearAlgebra.I(2), LinearAlgebra.I(2)), 2, 2, 2, 2)
    gate = Gate(mat, [Site(i), Site(j), Site(i; dual=true), Site(j; dual=true)])
    ψ = MPS([rand(2, 2), rand(2, 2, 2), rand(2, 2, 2), rand(2, 2, 2), rand(2, 2)])

    @test expect(ψ, gate) ≈ norm(ψ)^2
end

@testset "simple_update!" begin
    @testset "one site" begin
        i = 2
        mat = reshape(LinearAlgebra.I(2), 2, 2)
        gate = Gate(mat, [Site(i), Site(i; dual=true)])
        ψ = MPS([rand(2, 2), rand(2, 2, 2), rand(2, 2, 2), rand(2, 2, 2), rand(2, 2)])

        @testset "NonCanonical" begin
            ϕ = deepcopy(ψ)
            simple_update!(ϕ, gate; threshold=1e-14)
            @test length(tensors(ϕ)) == 5
            @test issetequal(size.(tensors(ϕ)), [(2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2)])
            @test isapprox(contract(ϕ), contract(ψ))
        end

        @testset "Canonical" begin
            ϕ = deepcopy(ψ)
            canonize!(ϕ)
            simple_update!(ϕ, gate; threshold=1e-14)
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
            simple_update!(ϕ, gate)

            @test length(tensors(ϕ)) == 4
            @test size(ϕ, inds(ϕ; bond=Bond(lane"2", lane"3"))) == 4
            @test isapprox(contract(ϕ), contract(ψ))
        end

        @testset "MixedCanonical" begin
            ϕ = simple_update!(canonize(ψ, lane"1"), gate)
            @test Tenet.checkform(ϕ)
            @test isapprox(contract(ϕ), contract(ψ))

            # `simple_update!` moves the orthogonality center where the gate is applied
            @test form(ϕ) == MixedCanonical([lane"2", lane"3"])
        end

        @testset "Canonical" begin
            ψ = MPS([rand(2, 2), rand(2, 2, 2), rand(2, 2, 2), rand(2, 2, 2), rand(2, 2)])
            normalize!(ψ)
            canonize!(ψ)

            let ϕ = simple_update!(deepcopy(ψ), gate)
                @test Tenet.checkform(ϕ)
                @test isapprox(contract(ϕ), contract(ψ))
            end

            let ϕ = simple_update!(deepcopy(ψ), gate; maxdim=1, normalize=true)
                @test norm(ϕ) ≈ 1.0
                @test Tenet.checkform(ϕ)
            end
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
            @test Tenet.checkform(ϕ_2)

            evolved = evolve!(deepcopy(canonize!(ψ)), mpo; maxdim=3)
            @test all(x -> x ≤ 3, vcat([collect(t) for t in vec(size.(tensors(evolved)))]...))
            @test form(evolved) == Canonical()
            @test Tenet.checkform(evolved)
        end

        @testset "MixedCanonical" begin
            mixed_canonize!(ϕ_3, lane"3")
            evolve!(ϕ_3, mpo)
            @test length(tensors(ϕ_3)) == 5
            @test form(ϕ_3) == MixedCanonical(lane"3")
            @test norm(ϕ_3) ≈ 1.0
            @test Tenet.checkform(ϕ_3)

            evolved = evolve!(deepcopy(mixed_canonize!(ψ, lane"3")), mpo; maxdim=3)
            @test all(x -> x ≤ 3, vcat([collect(t) for t in vec(size.(tensors(evolved)))]...))
            @test form(evolved) == MixedCanonical(lane"3")
            @test norm(evolved) ≈ 1.0
            @test Tenet.checkform(evolved)
        end

        t1 = contract(ϕ_1)
        t2 = contract(ϕ_2)
        t3 = contract(ϕ_3)

        @test t1 ≈ t2 ≈ t3
        @test only(overlap(ϕ_1, ϕ_2)) ≈ only(overlap(ϕ_1, ϕ_3)) ≈ only(overlap(ϕ_2, ϕ_3)) ≈ 1.0
    end
end

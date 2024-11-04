@testset_skip "Chain ansatz" begin
    @testset "Periodic boundary" begin
        @testset "State" begin
            qtn = Chain(State(), Periodic(), [rand(2, 4, 4) for _ in 1:3])
            @test socket(qtn) == State()
            @test nsites(qtn; set=:inputs) == 0
            @test nsites(qtn; set=:outputs) == 3
            @test issetequal(sites(qtn), [site"1", site"2", site"3"])
            @test boundary(qtn) == Periodic()
            @test leftindex(qtn, site"1") == rightindex(qtn, site"3") != nothing

            arrays = [rand(2, 1, 4), rand(2, 4, 3), rand(2, 3, 1)]
            qtn = Chain(State(), Periodic(), arrays) # Default order (:o, :l, :r)

            @test size(tensors(qtn; at=Site(1))) == (2, 1, 4)
            @test size(tensors(qtn; at=Site(2))) == (2, 4, 3)
            @test size(tensors(qtn; at=Site(3))) == (2, 3, 1)

            @test leftindex(qtn, Site(1)) == rightindex(qtn, Site(3))
            @test leftindex(qtn, Site(2)) == rightindex(qtn, Site(1))
            @test leftindex(qtn, Site(3)) == rightindex(qtn, Site(2))

            arrays = [permutedims(array, (3, 1, 2)) for array in arrays] # now we have (:r, :o, :l)
            qtn = Chain(State(), Periodic(), arrays; order=[:r, :o, :l])

            @test size(tensors(qtn; at=Site(1))) == (4, 2, 1)
            @test size(tensors(qtn; at=Site(2))) == (3, 2, 4)
            @test size(tensors(qtn; at=Site(3))) == (1, 2, 3)

            @test leftindex(qtn, Site(1)) == rightindex(qtn, Site(3))
            @test leftindex(qtn, Site(2)) == rightindex(qtn, Site(1))
            @test leftindex(qtn, Site(3)) == rightindex(qtn, Site(2))

            for i in 1:nsites(qtn)
                @test size(qtn, inds(qtn; at=Site(i))) == 2
            end
        end

        @testset "Operator" begin
            qtn = Chain(Operator(), Periodic(), [rand(2, 2, 4, 4) for _ in 1:3])
            @test socket(qtn) == Operator()
            @test nsites(qtn; set=:inputs) == 3
            @test nsites(qtn; set=:outputs) == 3
            @test issetequal(sites(qtn), [site"1", site"2", site"3", site"1'", site"2'", site"3'"])
            @test boundary(qtn) == Periodic()
            @test leftindex(qtn, site"1") == rightindex(qtn, site"3") != nothing

            arrays = [rand(2, 4, 1, 3), rand(2, 4, 3, 6), rand(2, 4, 6, 1)] # Default order (:o, :i, :l, :r)
            qtn = Chain(Operator(), Periodic(), arrays)

            @test size(tensors(qtn; at=Site(1))) == (2, 4, 1, 3)
            @test size(tensors(qtn; at=Site(2))) == (2, 4, 3, 6)
            @test size(tensors(qtn; at=Site(3))) == (2, 4, 6, 1)

            @test leftindex(qtn, Site(1)) == rightindex(qtn, Site(3))
            @test leftindex(qtn, Site(2)) == rightindex(qtn, Site(1))
            @test leftindex(qtn, Site(3)) == rightindex(qtn, Site(2))

            for i in 1:length(arrays)
                @test size(qtn, inds(qtn; at=Site(i))) == 2
                @test size(qtn, inds(qtn; at=Site(i; dual=true))) == 4
            end

            arrays = [permutedims(array, (4, 1, 3, 2)) for array in arrays] # now we have (:r, :o, :l, :i)
            qtn = Chain(Operator(), Periodic(), arrays; order=[:r, :o, :l, :i])

            @test size(tensors(qtn; at=Site(1))) == (3, 2, 1, 4)
            @test size(tensors(qtn; at=Site(2))) == (6, 2, 3, 4)
            @test size(tensors(qtn; at=Site(3))) == (1, 2, 6, 4)

            @test leftindex(qtn, Site(1)) == rightindex(qtn, Site(3)) !== nothing
            @test leftindex(qtn, Site(2)) == rightindex(qtn, Site(1)) !== nothing
            @test leftindex(qtn, Site(3)) == rightindex(qtn, Site(2)) !== nothing

            for i in 1:length(arrays)
                @test size(qtn, inds(qtn; at=Site(i))) == 2
                @test size(qtn, inds(qtn; at=Site(i; dual=true))) == 4
            end
        end
    end

    @testset "Open boundary" begin
        @testset "State" begin
            qtn = Chain(State(), Open(), [rand(2, 2), rand(2, 2, 2), rand(2, 2)])
            @test socket(qtn) == State()
            @test nsites(qtn; set=:inputs) == 0
            @test nsites(qtn; set=:outputs) == 3
            @test issetequal(sites(qtn), [site"1", site"2", site"3"])
            @test boundary(qtn) == Open()
            @test leftindex(qtn, site"1") == rightindex(qtn, site"3") == nothing

            arrays = [rand(2, 1), rand(2, 1, 3), rand(2, 3)]
            qtn = Chain(State(), Open(), arrays) # Default order (:o, :l, :r)

            @test size(tensors(qtn; at=Site(1))) == (2, 1)
            @test size(tensors(qtn; at=Site(2))) == (2, 1, 3)
            @test size(tensors(qtn; at=Site(3))) == (2, 3)

            @test leftindex(qtn, Site(1)) == rightindex(qtn, Site(3)) === nothing
            @test leftindex(qtn, Site(2)) == rightindex(qtn, Site(1))
            @test leftindex(qtn, Site(3)) == rightindex(qtn, Site(2))

            arrays = [permutedims(arrays[1], (2, 1)), permutedims(arrays[2], (3, 1, 2)), permutedims(arrays[3], (1, 2))] # now we have (:r, :o, :l)
            qtn = Chain(State(), Open(), arrays; order=[:r, :o, :l])

            @test size(tensors(qtn; at=Site(1))) == (1, 2)
            @test size(tensors(qtn; at=Site(2))) == (3, 2, 1)
            @test size(tensors(qtn; at=Site(3))) == (2, 3)

            @test leftindex(qtn, Site(1)) == rightindex(qtn, Site(3)) === nothing
            @test leftindex(qtn, Site(2)) == rightindex(qtn, Site(1)) !== nothing
            @test leftindex(qtn, Site(3)) == rightindex(qtn, Site(2)) !== nothing

            for i in 1:nsites(qtn)
                @test size(qtn, inds(qtn; at=Site(i))) == 2
            end
        end
        @testset "Operator" begin
            qtn = Chain(Operator(), Open(), [rand(2, 2, 4), rand(2, 2, 4, 4), rand(2, 2, 4)])
            @test socket(qtn) == Operator()
            @test nsites(qtn; set=:inputs) == 3
            @test nsites(qtn; set=:outputs) == 3
            @test issetequal(sites(qtn), [site"1", site"2", site"3", site"1'", site"2'", site"3'"])
            @test boundary(qtn) == Open()
            @test leftindex(qtn, site"1") == rightindex(qtn, site"3") == nothing

            arrays = [rand(2, 4, 1), rand(2, 4, 1, 3), rand(2, 4, 3)] # Default order (:o :i, :l, :r)
            qtn = Chain(Operator(), Open(), arrays)

            @test size(tensors(qtn; at=Site(1))) == (2, 4, 1)
            @test size(tensors(qtn; at=Site(2))) == (2, 4, 1, 3)
            @test size(tensors(qtn; at=Site(3))) == (2, 4, 3)

            @test leftindex(qtn, Site(1)) == rightindex(qtn, Site(3)) === nothing
            @test leftindex(qtn, Site(2)) == rightindex(qtn, Site(1)) !== nothing
            @test leftindex(qtn, Site(3)) == rightindex(qtn, Site(2)) !== nothing

            for i in 1:length(arrays)
                @test size(qtn, inds(qtn; at=Site(i))) == 2
                @test size(qtn, inds(qtn; at=Site(i; dual=true))) == 4
            end

            arrays = [
                permutedims(arrays[1], (3, 1, 2)),
                permutedims(arrays[2], (4, 1, 3, 2)),
                permutedims(arrays[3], (1, 3, 2)),
            ] # now we have (:r, :o, :l, :i)
            qtn = Chain(Operator(), Open(), arrays; order=[:r, :o, :l, :i])

            @test size(tensors(qtn; at=Site(1))) == (1, 2, 4)
            @test size(tensors(qtn; at=Site(2))) == (3, 2, 1, 4)
            @test size(tensors(qtn; at=Site(3))) == (2, 3, 4)

            @test leftindex(qtn, Site(1)) == rightindex(qtn, Site(3)) === nothing
            @test leftindex(qtn, Site(2)) == rightindex(qtn, Site(1)) !== nothing
            @test leftindex(qtn, Site(3)) == rightindex(qtn, Site(2)) !== nothing

            for i in 1:length(arrays)
                @test size(qtn, inds(qtn; at=Site(i))) == 2
                @test size(qtn, inds(qtn; at=Site(i; dual=true))) == 4
            end
        end
    end

    @testset "Site" begin
        using Tenet: leftsite, rightsite
        qtn = Chain(State(), Periodic(), [rand(2, 4, 4) for _ in 1:3])

        @test leftsite(qtn, Site(1)) == Site(3)
        @test leftsite(qtn, Site(2)) == Site(1)
        @test leftsite(qtn, Site(3)) == Site(2)

        @test rightsite(qtn, Site(1)) == Site(2)
        @test rightsite(qtn, Site(2)) == Site(3)
        @test rightsite(qtn, Site(3)) == Site(1)

        qtn = Chain(State(), Open(), [rand(2, 2), rand(2, 2, 2), rand(2, 2)])

        @test isnothing(leftsite(qtn, Site(1)))
        @test isnothing(rightsite(qtn, Site(3)))

        @test leftsite(qtn, Site(2)) == Site(1)
        @test leftsite(qtn, Site(3)) == Site(2)

        @test rightsite(qtn, Site(2)) == Site(3)
        @test rightsite(qtn, Site(1)) == Site(2)
    end

    @testset "truncate" begin
        qtn = Chain(State(), Open(), [rand(2, 2), rand(2, 2, 2), rand(2, 2)])
        canonize_site!(qtn, Site(2); direction=:right, method=:svd)

        @test_throws Tenet.MissingSchmidtCoefficientsException truncate!(qtn, [Site(1), Site(2)]; maxdim=1)
        # @test_throws ArgumentError truncate!(qtn, [Site(2), Site(3)])

        truncated = Tenet.truncate(qtn, [Site(2), Site(3)]; maxdim=1)
        @test size(truncated, rightindex(truncated, Site(2))) == 1
        @test size(truncated, leftindex(truncated, Site(3))) == 1

        singular_values = tensors(qtn; between=(Site(2), Site(3)))
        truncated = Tenet.truncate(qtn, [Site(2), Site(3)]; threshold=singular_values[2] + 0.1)
        @test size(truncated, rightindex(truncated, Site(2))) == 1
        @test size(truncated, leftindex(truncated, Site(3))) == 1
    end

    @testset "rand" begin
        using LinearAlgebra: norm

        @testset "State" begin
            n = 8
            χ = 10

            qtn = rand(Chain, Open, State; n, p=2, χ)
            @test socket(qtn) == State()
            @test nsites(qtn; set=:inputs) == 0
            @test nsites(qtn; set=:outputs) == n
            @test issetequal(sites(qtn), map(Site, 1:n))
            @test boundary(qtn) == Open()
            @test isapprox(norm(qtn), 1.0)
            @test maximum(last, size(qtn)) <= χ
        end

        @testset "Operator" begin
            n = 8
            χ = 10

            qtn = rand(Chain, Open, Operator; n, p=2, χ)
            @test socket(qtn) == Operator()
            @test nsites(qtn; set=:inputs) == n
            @test nsites(qtn; set=:outputs) == n
            @test issetequal(sites(qtn), vcat(map(Site, 1:n), map(adjoint ∘ Site, 1:n)))
            @test boundary(qtn) == Open()
            @test isapprox(norm(qtn), 1.0)
            @test maximum(last, size(qtn)) <= χ
        end
    end

    @testset "Canonization" begin
        using Tenet

        @testset "contract" begin
            qtn = rand(Chain, Open, State; n=5, p=2, χ=20)
            let canonized = canonize(qtn)
                @test_throws ArgumentError contract!(canonized; between=(Site(1), Site(2)), direction=:dummy)
            end

            canonized = canonize(qtn)

            for i in 1:4
                contract_some = contract(canonized; between=(Site(i), Site(i + 1)))
                Bᵢ = tensors(contract_some; at=Site(i))

                @test isapprox(contract(contract_some), contract(qtn))
                @test_throws ArgumentError tensors(contract_some; between=(Site(i), Site(i + 1)))

                @test isrightcanonical(contract_some, Site(i))
                @test isleftcanonical(
                    contract(canonized; between=(Site(i), Site(i + 1)), direction=:right), Site(i + 1)
                )

                Γᵢ = tensors(canonized; at=Site(i))
                Λᵢ₊₁ = tensors(canonized; between=(Site(i), Site(i + 1)))
                @test Bᵢ ≈ contract(Γᵢ, Λᵢ₊₁; dims=())
            end
        end

        @testset "canonize_site" begin
            qtn = Chain(State(), Open(), [rand(4, 4), rand(4, 4, 4), rand(4, 4)])

            @test_throws ArgumentError canonize_site!(qtn, Site(1); direction=:left)
            @test_throws ArgumentError canonize_site!(qtn, Site(3); direction=:right)

            for method in [:qr, :svd]
                canonized = canonize_site(qtn, site"1"; direction=:right, method=method)
                @test isleftcanonical(canonized, site"1")
                @test isapprox(contract(transform(TensorNetwork(canonized), Tenet.HyperFlatten())), contract(qtn))

                canonized = canonize_site(qtn, site"2"; direction=:right, method=method)
                @test isleftcanonical(canonized, site"2")
                @test isapprox(contract(transform(TensorNetwork(canonized), Tenet.HyperFlatten())), contract(qtn))

                canonized = canonize_site(qtn, site"2"; direction=:left, method=method)
                @test isrightcanonical(canonized, site"2")
                @test isapprox(contract(transform(TensorNetwork(canonized), Tenet.HyperFlatten())), contract(qtn))

                canonized = canonize_site(qtn, site"3"; direction=:left, method=method)
                @test isrightcanonical(canonized, site"3")
                @test isapprox(contract(transform(TensorNetwork(canonized), Tenet.HyperFlatten())), contract(qtn))
            end

            # Ensure that svd creates a new tensor
            @test length(tensors(canonize_site(qtn, Site(2); direction=:left, method=:svd))) == 4
        end

        @testset "canonize" begin
            using Tenet: isleftcanonical, isrightcanonical

            qtn = MPS([rand(4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4)])
            canonized = canonize(qtn)

            @test length(tensors(canonized)) == 9 # 5 tensors + 4 singular values vectors
            @test isapprox(contract(transform(TensorNetwork(canonized), Tenet.HyperFlatten())), contract(qtn))
            @test isapprox(norm(qtn), norm(canonized))

            # Extract the singular values between each adjacent pair of sites in the canonized chain
            Λ = [tensors(canonized; between=(Site(i), Site(i + 1))) for i in 1:4]
            @test map(λ -> sum(abs2, λ), Λ) ≈ ones(length(Λ)) * norm(canonized)^2

            for i in 1:5
                canonized = canonize(qtn)

                if i == 1
                    @test isleftcanonical(canonized, Site(i))
                elseif i == 5 # in the limits of the chain, we get the norm of the state
                    contract!(canonized; between=(Site(i - 1), Site(i)), direction=:right)
                    tensor = tensors(canonized; at=Site(i))
                    replace!(canonized, tensor => tensor / norm(canonized))
                    @test isleftcanonical(canonized, Site(i))
                else
                    contract!(canonized; between=(Site(i - 1), Site(i)), direction=:right)
                    @test isleftcanonical(canonized, Site(i))
                end
            end

            for i in 1:5
                canonized = canonize(qtn)

                if i == 1 # in the limits of the chain, we get the norm of the state
                    contract!(canonized; between=(Site(i), Site(i + 1)), direction=:left)
                    tensor = tensors(canonized; at=Site(i))
                    replace!(canonized, tensor => tensor / norm(canonized))
                    @test isrightcanonical(canonized, Site(i))
                elseif i == 5
                    @test isrightcanonical(canonized, Site(i))
                else
                    contract!(canonized; between=(Site(i), Site(i + 1)), direction=:left)
                    @test isrightcanonical(canonized, Site(i))
                end
            end
        end

        @testset "mixed_canonize" begin
            qtn = Chain(State(), Open(), [rand(4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4)])
            canonized = mixed_canonize(qtn, Site(3))

            @test length(tensors(canonized)) == length(tensors(qtn)) + 1

            @test isleftcanonical(canonized, Site(1))
            @test isleftcanonical(canonized, Site(2))
            @test isrightcanonical(canonized, Site(3))
            @test isrightcanonical(canonized, Site(4))
            @test isrightcanonical(canonized, Site(5))

            @test isapprox(contract(transform(TensorNetwork(canonized), Tenet.HyperFlatten())), contract(qtn))
        end
    end

    @test begin
        qtn = MPS([rand(4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4)])
        normalize!(qtn, Site(3))
        isapprox(norm(qtn), 1.0)
    end

    @testset "adjoint" begin
        qtn = rand(Chain, Open, State; n=5, p=2, χ=10)
        adjoint_qtn = adjoint(qtn)

        for i in 1:nsites(qtn)
            i < nsites(qtn) &&
                @test rightindex(adjoint_qtn, Site(i; dual=true)) == Symbol(String(rightindex(qtn, Site(i))) * "'")
            i > 1 && @test leftindex(adjoint_qtn, Site(i; dual=true)) == Symbol(String(leftindex(qtn, Site(i))) * "'")
        end

        @test isapprox(contract(qtn), contract(adjoint_qtn))
    end

    @testset "evolve!" begin
        @testset "one site" begin
            i = 2
            mat = reshape(LinearAlgebra.I(2), 2, 2)
            gate = Dense(Tenet.Operator(), mat; sites=[Site(i), Site(i; dual=true)])

            qtn = Chain(State(), Open(), [rand(2, 2), rand(2, 2, 2), rand(2, 2, 2), rand(2, 2, 2), rand(2, 2)])

            @testset "canonical form" begin
                canonized = canonize(qtn)

                evolved = evolve!(deepcopy(canonized), gate; threshold=1e-14)
                @test isapprox(contract(evolved), contract(canonized))
                @test issetequal(size.(tensors(evolved)), [(2, 2), (2,), (2, 2, 2), (2,), (2, 2, 2), (2,), (2, 2)])
                @test isapprox(contract(evolved), contract(qtn))
            end

            @testset "arbitrary chain" begin
                evolved = evolve!(deepcopy(qtn), gate; threshold=1e-14, iscanonical=false)
                @test length(tensors(evolved)) == 5
                @test issetequal(size.(tensors(evolved)), [(2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2)])
                @test isapprox(contract(evolved), contract(qtn))
            end
        end

        @testset "two sites" begin
            i, j = 2, 3
            mat = reshape(kron(LinearAlgebra.I(2), LinearAlgebra.I(2)), 2, 2, 2, 2)
            gate = Dense(Tenet.Operator(), mat; sites=[Site(i), Site(j), Site(i; dual=true), Site(j; dual=true)])

            qtn = Chain(State(), Open(), [rand(2, 2), rand(2, 2, 2), rand(2, 2, 2), rand(2, 2)])

            @testset "canonical form" begin
                canonized = canonize(qtn)

                evolved = evolve!(deepcopy(canonized), gate; threshold=1e-14)
                @test isapprox(contract(evolved), contract(canonized))
                @test issetequal(size.(tensors(evolved)), [(2, 2), (2,), (2, 2, 2), (2,), (2, 2, 2), (2,), (2, 2)])
                @test isapprox(contract(evolved), contract(qtn))
            end

            @testset "arbitrary chain" begin
                evolved = evolve!(deepcopy(qtn), gate; threshold=1e-14, iscanonical=false)
                @test length(tensors(evolved)) == 5
                @test issetequal(size.(tensors(evolved)), [(2, 2), (2, 2, 2), (2,), (2, 2, 2), (2, 2, 2), (2, 2)])
                @test isapprox(contract(evolved), contract(qtn))
            end
        end
    end

    @testset "expect" begin
        i, j = 2, 3
        mat = reshape(kron(LinearAlgebra.I(2), LinearAlgebra.I(2)), 2, 2, 2, 2)
        gate = Dense(Tenet.Operator(), mat; sites=[Site(i), Site(j), Site(i; dual=true), Site(j; dual=true)])

        qtn = Chain(State(), Open(), [rand(2, 2), rand(2, 2, 2), rand(2, 2, 2), rand(2, 2, 2), rand(2, 2)])

        @test isapprox(expect(qtn, [gate]), norm(qtn)^2)
    end
end

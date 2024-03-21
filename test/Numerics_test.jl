@testset "Numerics" begin
    using LinearAlgebra

    @testset "basic arithmetic" begin
        A = Tensor(rand(2, 3, 2, 4), (:i, :j, :k, :l))
        B = Tensor(rand(3, 4, 2, 2), (:j, :l, :i, :k))

        C = A + B
        @test issetequal(inds(C), (:i, :j, :k, :l))
        @test issetequal(size(C), (2, 3, 2, 4))
        @test parent(C) ≈ parent(A) + permutedims(parent(B), (3, 1, 4, 2))

        D = A - B
        @test issetequal(inds(D), (:i, :j, :k, :l))
        @test issetequal(size(D), (2, 3, 2, 4))
        @test parent(D) ≈ parent(A) - permutedims(parent(B), (3, 1, 4, 2))
    end

    @testset "contract" begin
        @testset "axis sum" begin
            A = Tensor(rand(2, 3, 4), (:i, :j, :k))

            C = contract(A, dims = (:i,))
            C_ein = ein"ijk -> jk"(A)
            @test inds(C) == [:j, :k]
            @test size(C) == size(C_ein) == (3, 4)
            @test parent(C) ≈ C_ein
        end

        @testset "diagonal" begin
            A = Tensor(rand(2, 3, 2), (:i, :j, :i))

            C = contract(A, dims = ())
            C_ein = ein"iji -> ij"(A)
            @test inds(C) == [:i, :j]
            @test size(C) == size(C_ein) == (2, 3)
            @test parent(C) ≈ C_ein
        end

        @testset "trace" begin
            A = Tensor(rand(2, 3, 2), (:i, :j, :i))

            C = contract(A, dims = (:i,))
            C_ein = ein"iji -> j"(A)
            @test inds(C) == [:j]
            @test size(C) == size(C_ein) == (3,)
            @test parent(C) ≈ C_ein
        end

        @testset "matrix multiplication" begin
            A = Tensor(rand(2, 3), (:i, :j))
            B = Tensor(rand(3, 4), (:j, :k))

            C = contract(A, B)
            C_mat = parent(A) * parent(B)
            @test inds(C) == [:i, :k]
            @test size(C) == (2, 4) == size(C_mat)
            @test parent(C) ≈ parent(A * B) ≈ C_mat
        end

        @testset "inner product" begin
            A = Tensor(rand(3, 4), (:i, :j))
            B = Tensor(rand(4, 3), (:j, :i))

            C = contract(A, B)
            C_res = LinearAlgebra.tr(parent(A) * parent(B))
            @test inds(C) == Symbol[]
            @test size(C) == () == size(C_res)
            @test only(C) ≈ C_res
        end

        @testset "outer product" begin
            A = Tensor(rand(2, 2), (:i, :j))
            B = Tensor(rand(2, 2), (:k, :l))

            C = contract(A, B)
            C_ein = ein"ij, kl -> ijkl"(A, B)
            @test size(C) == (2, 2, 2, 2) == size(C_ein)
            @test inds(C) == [:i, :j, :k, :l]
            @test parent(C) ≈ C_ein
        end

        @testset "scale" begin
            A = Tensor(rand(2, 2), (:i, :j))
            scalar = 2.0

            C = contract(A, scalar)
            @test inds(C) == [:i, :j]
            @test size(C) == (2, 2)
            @test parent(C) ≈ parent(A) * scalar

            D = contract(scalar, A)
            @test inds(D) == [:i, :j]
            @test size(D) == (2, 2)
            @test parent(D) ≈ scalar * parent(A)
        end

        @testset "manual" begin
            A = Tensor(rand(2, 3, 4), (:i, :j, :k))
            B = Tensor(rand(4, 5, 3), (:k, :l, :j))

            # Contraction of all common indices
            C = contract(A, B, dims = (:j, :k))
            C_ein = ein"ijk, klj -> il"(A, B)
            @test inds(C) == [:i, :l]
            @test size(C) == (2, 5) == size(C_ein)
            @test parent(C) ≈ C_ein

            # Contraction of not all common indices
            C = contract(A, B, dims = (:j,))
            C_ein = ein"ijk, klj -> ikl"(A, B)
            @test inds(C) == [:i, :k, :l]
            @test size(C) == (2, 4, 5) == size(C_ein)
            @test parent(C) ≈ C_ein

            @testset "Complex numbers" begin
                A = Tensor(rand(Complex{Float64}, 2, 3, 4), (:i, :j, :k))
                B = Tensor(rand(Complex{Float64}, 4, 5, 3), (:k, :l, :j))

                C = contract(A, B, dims = (:j, :k))
                C_ein = ein"ijk, klj -> il"(A, B)
                @test inds(C) == [:i, :l]
                @test size(C) == (2, 5) == size(C_ein)
                @test parent(C) ≈ C_ein
            end
        end

        @testset "multiple tensors" begin
            A = Tensor(rand(2, 3, 4), (:i, :j, :k))
            B = Tensor(rand(4, 5, 3), (:k, :l, :j))
            C = Tensor(rand(5, 6, 2), (:l, :m, :i))
            D = Tensor(rand(6, 7, 2), (:m, :n, :i))

            contracted = contract(A, B, C, D)
            @test issetequal(inds(contracted), (:n, :i))
            @test issetequal(size(contracted), (7, 2))
            @test contracted ≈ contract(contract(contract(A, B), C), D)
        end
    end

    @testset "svd" begin
        data = rand(ComplexF64, 2, 4, 6, 8)
        tensor = Tensor(data, (:i, :j, :k, :l))

        # throw if left_inds is not provided
        @test_throws ArgumentError svd(tensor)

        # throw if index is not present
        @test_throws ArgumentError svd(tensor, left_inds = [:z])
        @test_throws ArgumentError svd(tensor, right_inds = [:z])

        # throw if no inds left
        @test_throws ArgumentError svd(tensor, left_inds = (:i, :j, :k, :l))
        @test_throws ArgumentError svd(tensor, right_inds = (:i, :j, :k, :l))

        # throw if chosen virtual index already present
        @test_throws ArgumentError svd(tensor, left_inds = (:i,), virtualind = :j)

        U, s, V = svd(tensor, left_inds = [:i, :j], virtualind = :x)

        @test inds(U) == [:i, :j, :x]
        @test inds(s) == [:x]
        @test inds(V) == [:k, :l, :x]

        @test size(U) == (2, 4, 8)
        @test size(s) == (8,)
        @test size(V) == (6, 8, 8)

        @test isapprox(contract(contract(U, s, dims = Symbol[]), V), tensor)
    end

    @testset "qr" begin
        data = rand(2, 4, 6, 8)
        tensor = Tensor(data, (:i, :j, :k, :l))
        vidx = :x

        # throw if left_inds is not provided
        @test_throws ArgumentError qr(tensor)

        # throw if index is not present
        @test_throws ArgumentError qr(tensor, left_inds = [:z])
        @test_throws ArgumentError qr(tensor, right_inds = [:z])

        # throw if no inds left
        @test_throws ArgumentError qr(tensor, left_inds = (:i, :j, :k, :l))
        @test_throws ArgumentError qr(tensor, right_inds = (:i, :j, :k, :l))

        # throw if chosen virtual index already present
        @test_throws ArgumentError qr(tensor, left_inds = (:i,), virtualind = :j)

        Q, R = qr(tensor, left_inds = (:i, :j), virtualind = vidx)

        @test inds(Q) == [:i, :j, :x]
        @test inds(R) == [:x, :k, :l]

        @test size(Q) == (2, 4, 8)
        @test size(R) == (8, 6, 8)

        @test isapprox(contract(Q, R), tensor)
    end

    @testset "lu" begin
        data = rand(2, 4, 6, 8)
        tensor = Tensor(data, (:i, :j, :k, :l))
        vidx = [:x, :y]

        # throw if no index is provided
        @test_throws ArgumentError lu(tensor)

        # throw if index is not present
        @test_throws ArgumentError lu(tensor, left_inds = (:z,))
        @test_throws ArgumentError lu(tensor, right_inds = (:z,))

        # throw if no inds left
        @test_throws ArgumentError lu(tensor, left_inds = (:i, :j, :k, :l))
        @test_throws ArgumentError lu(tensor, right_inds = (:i, :j, :k, :l))

        # throw if chosen virtual index already present
        @test_throws ArgumentError qr(tensor, left_inds = (:i,), virtualind = :j)

        L, U, P = lu(tensor, left_inds = [:i, :j], virtualind = vidx)
        @test inds(L) == [:x, :y]
        @test inds(U) == [:y, :k, :l]
        @test inds(P) == [:i, :j, :x]

        @test size(L) == (8, 8)
        @test size(U) == (8, 6, 8)
        @test size(P) == (2, 4, 8)

        @test isapprox(contract(L, U, P), tensor)
    end
end

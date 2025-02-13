
using Tenet
using Tenet: TensorOperationsBackend
using LinearAlgebra
using TensorOperations: TensorOperations
using OMEinsum: @ein_str

@testset "contract" begin
    # NOTE pattern not supported by TensorOperations
    # @testset "axis sum" begin
    #     A = Tensor(rand(2, 3, 4), (:i, :j, :k))

    #     C = contract(A; dims=(:i,))
    #     C_ein = ein"ijk -> jk"(parent(A))
    #     @test inds(C) == (:j, :k)
    #     @test size(C) == size(C_ein) == (3, 4)
    #     @test parent(C) ≈ C_ein
    # end

    # NOTE pattern not supported by TensorOperations
    # @testset "diagonal" begin
    #     A = Tensor(rand(2, 3, 2), (:i, :j, :i))

    #     C = contract(A; dims=())
    #     C_ein = ein"iji -> ij"(parent(A))
    #     @test inds(C) == (:i, :j)
    #     @test size(C) == size(C_ein) == (2, 3)
    #     @test parent(C) ≈ C_ein
    # end

    @testset "trace" begin
        A = Tensor(rand(2, 3, 2), (:i, :j, :i))

        C = contract(TensorOperationsBackend(), A; dims=(:i,))
        C_ein = ein"iji -> j"(parent(A))
        @test inds(C) == (:j,)
        @test size(C) == size(C_ein) == (3,)
        @test parent(C) ≈ C_ein
    end

    @testset "matrix multiplication" begin
        A = Tensor(rand(2, 3), (:i, :j))
        B = Tensor(rand(3, 4), (:j, :k))

        C = contract(TensorOperationsBackend(), A, B)
        C_mat = parent(A) * parent(B)
        @test inds(C) == (:i, :k)
        @test size(C) == (2, 4) == size(C_mat)
        @test parent(C) ≈ parent(A * B) ≈ C_mat
    end

    @testset "inner product" begin
        A = Tensor(rand(3, 4), (:i, :j))
        B = Tensor(rand(4, 3), (:j, :i))

        C = contract(TensorOperationsBackend(), A, B)
        C_res = LinearAlgebra.tr(parent(A) * parent(B))
        @test inds(C) == ()
        @test size(C) == () == size(C_res)
        @test only(C) ≈ C_res
    end

    @testset "outer product" begin
        A = Tensor(rand(2, 2), (:i, :j))
        B = Tensor(rand(2, 2), (:k, :l))

        C = contract(TensorOperationsBackend(), A, B)
        C_ein = ein"ij, kl -> ijkl"(parent(A), parent(B))
        @test size(C) == (2, 2, 2, 2) == size(C_ein)
        @test inds(C) == (:i, :j, :k, :l)
        @test parent(C) ≈ C_ein
    end

    @testset "manual" begin
        A = Tensor(rand(2, 3, 4), (:i, :j, :k))
        B = Tensor(rand(4, 5, 3), (:k, :l, :j))

        # Contraction of all common indices
        C = contract(TensorOperationsBackend(), A, B; dims=(:j, :k))
        C_ein = ein"ijk, klj -> il"(parent(A), parent(B))
        @test inds(C) == (:i, :l)
        @test size(C) == (2, 5) == size(C_ein)
        @test parent(C) ≈ C_ein

        # Contraction of not all common indices
        # NOTE pattern not supported by TensorOperations
        # C = contract(TensorOperationsBackend(), A, B; dims=(:j,))
        # C_ein = ein"ijk, klj -> ikl"(parent(A), parent(B))
        # @test inds(C) == (:i, :k, :l)
        # @test size(C) == (2, 4, 5) == size(C_ein)
        # @test parent(C) ≈ C_ein

        @testset "Complex numbers" begin
            A = Tensor(rand(Complex{Float64}, 2, 3, 4), (:i, :j, :k))
            B = Tensor(rand(Complex{Float64}, 4, 5, 3), (:k, :l, :j))

            C = contract(TensorOperationsBackend(), A, B; dims=(:j, :k))
            C_ein = ein"ijk, klj -> il"(parent(A), parent(B))
            @test inds(C) == (:i, :l)
            @test size(C) == (2, 5) == size(C_ein)
            @test parent(C) ≈ C_ein
        end
    end

    # NOTE pattern not supported by TensorOperations
    # @testset "multiple tensors" begin
    #     A = Tensor(rand(2, 3, 4), (:i, :j, :k))
    #     B = Tensor(rand(4, 5, 3), (:k, :l, :j))
    #     C = Tensor(rand(5, 6, 2), (:l, :m, :i))
    #     D = Tensor(rand(6, 7, 2), (:m, :n, :i))

    #     contracted = contract(TensorOperationsBackend(), A, B, C, D)
    #     @test issetequal(inds(contracted), (:n, :i))
    #     @test issetequal(size(contracted), (7, 2))
    #     @test contracted ≈ contract(
    #         TensorOperationsBackend(),
    #         contract(TensorOperationsBackend(), contract(TensorOperationsBackend(), A, B), C),
    #         D,
    #     )
    # end
end

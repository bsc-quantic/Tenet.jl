@testset "KrylovKit.eigsolve" begin
    using Tenet: Tensor
    using KrylovKit

    A = rand(ComplexF64, 4, 4)
    data = (A + A') / 2 # Make it Hermitian
    tensor = Tensor(data, (:i, :j))

    # throw if the resulting matrix is not square
    tensor_non_square = Tensor(rand(ComplexF64, 2, 4, 6), (:i, :j, :k))
    @test_throws ArgumentError eigsolve(tensor_non_square; left_inds=[:i, :j])
    @test_throws ArgumentError eigsolve(tensor_non_square; right_inds=[:j, :k])

    # throw if index is not present
    @test_throws ArgumentError eigsolve(tensor, left_inds=[:z])
    @test_throws ArgumentError eigsolve(tensor, right_inds=[:z])

    # Perform eigensolve
    vals, vecs = eigsolve(tensor; left_inds=[:i], right_inds=[:j])

    @test length(vals) == 4
    @test length(vecs) == 4

    for vec in vecs
        @test inds(vec) == [:i]
        @test size(vec) == (4,)
    end

    # Convert vecs to matrix form for reconstruction
    V_matrix = hcat([reshape(parent(vec), :) for vec in vecs]...)
    D_matrix = Diagonal(vals)
    reconstructed_matrix = V_matrix * D_matrix * inv(V_matrix)

    # Ensure the reconstruction is correct
    reconstructed_tensor = Tensor(reconstructed_matrix, (:i, :j))
    @test isapprox(reconstructed_tensor, tensor)

    # Test consistency with permuted tensor
    tensor_permuted = Tensor(data, (:j, :i))

    vals_perm, vecs_perm = eigsolve(tensor_permuted; left_inds=[:j], right_inds=[:i])

    @test length(vals_perm) == 4
    @test length(vecs_perm) == 4

    # Ensure the eigenvalues are the same
    @test isapprox(sort(real.(vals)), sort(real.(vals_perm))) && isapprox(sort(imag.(vals)), sort(imag.(vals_perm)))

    V_matrix_perm = hcat([reshape(parent(vec), :) for vec in vecs_perm]...)
    D_matrix_perm = Diagonal(vals)
    reconstructed_matrix_perm = V_matrix_perm * D_matrix_perm * inv(V_matrix_perm)

    # Ensure the reconstruction is correct
    reconstructed_tensor_perm = Tensor(reconstructed_matrix_perm, (:j, :i))
    @test isapprox(reconstructed_tensor_perm, tensor_permuted)

    @test parent(reconstructed_tensor) ≈ parent(reconstructed_tensor_perm)

    @testset "Lanczos" begin
        A = rand(ComplexF64, 4, 4)
        data = (A + A') / 2 # Make it Hermitian
        tensor = Tensor(data, (:i, :j))

        vals, vecs = eigsolve(
            tensor, rand(ComplexF64, 4), 1, :SR, Lanczos(; krylovdim=2, tol=1e-16); left_inds=[:i], right_inds=[:j]
        )

        @test length(vals) == 1
        @test length(vecs) == 1
    end
end

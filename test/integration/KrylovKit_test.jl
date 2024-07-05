@testset "KrylovKit.eigsolve" begin
    using Tenet: Tensor
    using KrylovKit

    A = rand(ComplexF64, 4, 4)
    data = (A + A') / 2 # Make it Hermitian
    tensor = Tensor(data, (:i, :j))

    # Perform eigensolve
    vals, vecs, info = eigsolve(tensor; left_inds=[:i], right_inds=[:j])

    @test length(vals) == 4
    @test length(vecs) == 4

    for vec in vecs
        @test inds(vec) == [:i]
        @test size(vec) == (4,)
    end

    # throw if index is not present
    @test_throws ArgumentError eigsolve(tensor; left_inds=[:z])
    @test_throws ArgumentError eigsolve(tensor; right_inds=[:z])

    # throw if the resulting matrix is not square
    tensor_non_square = Tensor(rand(ComplexF64, 2, 4, 6), (:i, :j, :k))
    @test_throws ArgumentError eigsolve(tensor_non_square; left_inds=[:i, :j], right_inds=[:k])
    @test_throws ArgumentError eigsolve(tensor_non_square; right_inds=[:j, :k])

    # Convert vecs to matrix form for reconstruction
    V_matrix = hcat([reshape(parent(vec), :) for vec in vecs]...)
    D_matrix = Diagonal(vals)
    reconstructed_matrix = V_matrix * D_matrix * inv(V_matrix)

    # Ensure the reconstruction is correct
    reconstructed_tensor = Tensor(reconstructed_matrix, (:i, :j))
    @test isapprox(reconstructed_tensor, tensor)

    # Test consistency with permuted tensor
    vals_perm, vecs_perm, info = eigsolve(tensor; left_inds=[:j], right_inds=[:i])

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

    @test parent(reconstructed_tensor) ≈ parent(transpose(reconstructed_tensor_perm))

    @testset "Lanczos" begin
        vals_lanczos, vecs_lanczos = eigsolve(
            tensor, rand(ComplexF64, 4), 1, :SR, Lanczos(; krylovdim=2, tol=1e-16); left_inds=[:i], right_inds=[:j]
        )

        @test length(vals_lanczos) == 1
        @test length(vecs_lanczos) == 1

        @test minimum(vals) ≈ first(vals_lanczos)
    end

    A = rand(ComplexF64, 4, 4)
    data = (A + A') / 2 # Make it Hermitian
    tensor = Tensor(reshape(data, 2, 2, 2, 2), (:i, :j, :k, :l))

    vals, vecs, info = eigsolve(tensor; left_inds=[:i, :j], right_inds=[:k, :l])

    # Convert vecs to matrix form for reconstruction
    V_matrix = hcat([reshape(parent(vec), :) for vec in vecs]...)
    D_matrix = Diagonal(vals)
    reconstructed_matrix = V_matrix * D_matrix * inv(V_matrix)

    @test isapprox(reconstructed_matrix, data)
end

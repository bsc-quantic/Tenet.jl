@testset "Reactant" begin
    using Reactant

    # TODO test `make_tracer`
    # TODO test `create_result`
    # TODO test `traced_getfield`
    # TODO test `Enzyme.autodiff`

    # TODO test unary einsum
    # TODO test scalar × tensor
    # TODO test inner product, outer product, manual and multiple tensors binary einsums
    @testset "contract" begin
        @testset "matrix multiplication" begin
            A = Tensor(rand(2, 3), (:i, :j))
            B = Tensor(rand(3, 4), (:j, :k))
            C = contract(A, B)

            Are = adapt(ConcreteRArray, A)
            Bre = adapt(ConcreteRArray, B)
            Cre = @jit contract(Are, Bre)
            @test inds(Cre) == inds(C)
            @test size(Cre) == size(C)
            @test parent(Cre) ≈ parent(C)

            f(a, b) = contract(a, b; out=[:k, :i])
            Cre = @jit f(Are, Bre)
            @test inds(C) == reverse(inds(Cre))
            @test size(C) == reverse(size(Cre))
            @test parent(C) ≈ transpose(parent(Cre))
        end
    end
end
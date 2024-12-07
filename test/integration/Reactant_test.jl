@testset "Reactant" begin
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

            A = Tensor(rand(2, 3), (:i, :j))
            B = Tensor(rand(4, 3), (:j, :k))
            Are = adapt(ConcreteRArray, A)
            Bre = adapt(ConcreteRArray, B)

            f(a, b) = contract(a, b; out=[:k, :i])
            C = f(A, B)
            Cre = @jit f(Are, Bre)

            @test inds(Cre) == inds(C)
            @test size(Cre) == size(C)
            @test parent(Cre) ≈ parent(C)
        end
    end
end

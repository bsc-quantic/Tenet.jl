@testset "Reactant" begin
    # TODO test `make_tracer`
    # TODO test `create_result`
    # TODO test `traced_getfield`
    # TODO test `Enzyme.autodiff`

    # TODO test unary einsum
    # TODO test scalar × tensor
    @testset "contract" begin
        @testset "matrix multiplication" begin
            A = Tensor(rand(2, 3), (:i, :j))
            B = Tensor(rand(3, 4), (:j, :k))

            @testset "without permutation" begin
                C = contract(A, B)
                Are = adapt(ConcreteRArray, A)
                Bre = adapt(ConcreteRArray, B)
                Cre = @jit contract(Are, Bre)

                @test inds(Cre) == inds(C)
                @test size(Cre) == size(C)
                @test parent(Cre) ≈ parent(C)

                @test "hybrid" begin
                    Cre = @jit contract(A, Bre)

                    @test inds(Cre) == inds(C)
                    @test size(Cre) == size(C)
                    @test parent(Cre) ≈ parent(C)
                end
            end

            @testset "with permutation" begin
                f(a, b) = contract(a, b; out=[:k, :i])
                C = f(A, B)
                Cre = @jit f(Are, Bre)

                @test inds(Cre) == inds(C)
                @test size(Cre) == size(C)
                @test parent(Cre) ≈ parent(C)

                @test "hybrid" begin
                    Cre = @jit f(A, Bre)

                    @test inds(Cre) == inds(C)
                    @test size(Cre) == size(C)
                    @test parent(Cre) ≈ parent(C)
                end
            end
        end

        @testset "inner product" begin
            A = Tensor(rand(3, 4), (:i, :j))
            B = Tensor(rand(4, 3), (:j, :i))
            C = contract(A, B)

            Are = adapt(ConcreteRArray, A)
            Bre = adapt(ConcreteRArray, B)
            Cre = @jit contract(Are, Bre)

            @test inds(Cre) == inds(C)
            @test size(Cre) == size(C)
            @test only(Cre) ≈ only(C)
        end

        @testset "outer product" begin
            A = Tensor(rand(2, 2), (:i, :j))
            B = Tensor(rand(2, 2), (:k, :l))
            C = contract(A, B)

            Are = adapt(ConcreteRArray, A)
            Bre = adapt(ConcreteRArray, B)
            Cre = @jit contract(Are, Bre)

            @test inds(Cre) == inds(C)
            @test size(Cre) == size(C)
            @test adapt(Array, Cre) ≈ C
        end

        @testset "manual" begin
            A = Tensor(rand(2, 3, 4), (:i, :j, :k))
            B = Tensor(rand(4, 5, 3), (:k, :l, :j))
            Are = adapt(ConcreteRArray, A)
            Bre = adapt(ConcreteRArray, B)

            # Contraction of all common indices
            f1(a, b) = contract(a, b; dims=(:j, :k))
            C = f1(A, B)
            Cre = @jit f1(Are, Bre)

            @test inds(Cre) == inds(C)
            @test size(Cre) == size(C)
            @test adapt(Array, Cre) ≈ C

            # Contraction of not all common indices
            f2(a, b) = contract(a, b; dims=(:j,))
            C = f2(A, B)
            Cre = @jit f2(Are, Bre)

            @test inds(Cre) == inds(C)
            @test size(Cre) == size(C)
            @test adapt(Array, Cre) ≈ C

            @testset "Complex numbers" begin
                A = Tensor(rand(Complex{Float64}, 2, 3, 4), (:i, :j, :k))
                B = Tensor(rand(Complex{Float64}, 4, 5, 3), (:k, :l, :j))
                Are = adapt(ConcreteRArray, A)
                Bre = adapt(ConcreteRArray, B)

                C = f1(A, B)
                Cre = @jit f1(Are, Bre)

                @test inds(Cre) == inds(C)
                @test size(Cre) == size(C)
                @test adapt(Array, Cre) ≈ C
            end
        end

        @testset "multiple tensors" begin
            A = Tensor(rand(2, 3, 4), (:i, :j, :k))
            B = Tensor(rand(4, 5, 3), (:k, :l, :j))
            C = Tensor(rand(5, 6, 2), (:l, :m, :i))
            D = Tensor(rand(6, 7, 2), (:m, :n, :i))

            Are = adapt(ConcreteRArray, A)
            Bre = adapt(ConcreteRArray, B)
            Cre = adapt(ConcreteRArray, C)
            Dre = adapt(ConcreteRArray, D)

            X = contract(A, B, C, D)
            Xre = @jit contract(Are, Bre, Cre, Dre)

            @test inds(Xre) == inds(X)
            @test size(Xre) == size(X)
            @test adapt(Array, Xre) ≈ X
        end
    end
end

@testset "MatrixProductOperator" begin
    using Tenet: TensorNetwork, Operator, Closed, Open, bounds, MatrixProductOperator

    @testset "Types" begin
        @test MatrixProductOperator <: Operator
        @test all(T -> MatrixProductOperator{T} <: Operator, [Open, Closed])
        @test all(T -> MatrixProductOperator{T} <: Operator{T}, [Open, Closed])
        @test all(B -> bounds(MatrixProductOperator{B}) == B, [Open, Closed])
    end

    @testset "Constructor" begin
        @test_skip begin
            arrays = [rand(2, 2)]
            MatrixProductOperator(arrays)
        end

        @test begin
            arrays = [rand(2, 1, 1), rand(2, 1, 1)]
            MatrixProductOperator(arrays) isa TensorNetwork{MatrixProductOperator{Open}}
        end

        @testset "`Open` boundary" begin
            # product operator
            @test begin
                arrays = [rand(1, 2, 2), rand(1, 1, 2, 2), rand(1, 2, 2)]
                MatrixProductOperator{Open}(arrays) isa TensorNetwork{MatrixProductOperator{Open}}
            end

            # alternative constructor
            @test begin
                arrays = [rand(1, 2, 2), rand(1, 1, 2, 2), rand(1, 2, 2)]
                MatrixProductOperator(arrays; bounds = Open) isa TensorNetwork{MatrixProductOperator{Open}}
            end

            # entangling operator
            @test begin
                i = 3
                o = 5
                arrays = [rand(2, i, o), rand(2, 4, i, o), rand(4, i, o)]
                MatrixProductOperator{Open}(arrays) isa TensorNetwork{MatrixProductOperator{Open}}
            end

            # entangling operator - change order
            @test begin
                i = 3
                o = 5
                arrays = [rand(i, 2, o), rand(2, i, 4, o), rand(4, i, o)]
                MatrixProductOperator{Open}(arrays, order = (:l, :i, :r, :o)) isa
                TensorNetwork{MatrixProductOperator{Open}}
            end

            # fail on Open with Closed format
            @test_throws DimensionMismatch begin
                arrays = [rand(1, 1, 2, 2), rand(1, 1, 2, 2), rand(1, 1, 2, 2)]
                MatrixProductOperator{Open}(arrays) isa TensorNetwork{MatrixProductOperator{Open}}
            end

            @testset "Metadata" begin
                @testset "alias" begin
                    arrays = [rand(1, 2, 2), rand(1, 1, 2, 2), rand(1, 2, 2)]
                    ψ = MatrixProductOperator{Open}(arrays, order = (:l, :r, :i, :o))

                    @test ψ.meta[:order] == Dict(:l => 1, :r => 2, :i => 3, :o => 4)

                    @test issetequal(keys(tensors(ψ, 1).meta[:alias]), [:r, :i, :o])
                    @test issetequal(keys(tensors(ψ, 2).meta[:alias]), [:l, :r, :i, :o])
                    @test issetequal(keys(tensors(ψ, 3).meta[:alias]), [:l, :i, :o])

                    @test tensors(ψ, 1).meta[:alias][:r] === tensors(ψ, 2).meta[:alias][:l]
                    @test tensors(ψ, 2).meta[:alias][:r] === tensors(ψ, 3).meta[:alias][:l]
                end
            end
        end

        @testset "`Closed` boundary" begin
            # product operator
            @test begin
                arrays = [rand(1, 1, 2, 2), rand(1, 1, 2, 2), rand(1, 1, 2, 2)]
                MatrixProductOperator{Closed}(arrays) isa TensorNetwork{MatrixProductOperator{Closed}}
            end

            # alternative constructor
            @test begin
                arrays = [rand(1, 1, 2, 2), rand(1, 1, 2, 2), rand(1, 1, 2, 2)]
                MatrixProductOperator(arrays; bounds = Closed) isa TensorNetwork{MatrixProductOperator{Closed}}
            end

            # entangling operator
            @test begin
                i = 3
                o = 5
                arrays = [rand(2, 4, i, o), rand(4, 8, i, o), rand(8, 2, i, o)]
                MatrixProductOperator{Closed}(arrays) isa TensorNetwork{MatrixProductOperator{Closed}}
            end

            # entangling operator - change order
            @test begin
                i = 3
                o = 5
                arrays = [rand(2, i, 4, o), rand(4, i, 8, o), rand(8, i, 2, o)]
                MatrixProductOperator{Closed}(arrays, order = (:l, :i, :r, :o)) isa
                TensorNetwork{MatrixProductOperator{Closed}}
            end

            # fail on Closed with Open format
            @test_throws DimensionMismatch begin
                arrays = [rand(1, 2, 2), rand(1, 1, 2, 2), rand(1, 2, 2)]
                MatrixProductOperator{Closed}(arrays) isa TensorNetwork{MatrixProductOperator{Closed}}
            end

            @testset "Metadata" begin
                @testset "alias" begin
                    arrays = [rand(1, 1, 2, 2), rand(1, 1, 2, 2), rand(1, 1, 2, 2)]
                    ψ = MatrixProductOperator{Closed}(arrays, order = (:l, :r, :i, :o))

                    @test ψ.meta[:order] == Dict(:l => 1, :r => 2, :i => 3, :o => 4)

                    @test issetequal(keys(tensors(ψ, 1).meta[:alias]), [:l, :r, :i, :o])
                    @test issetequal(keys(tensors(ψ, 2).meta[:alias]), [:l, :r, :i, :o])
                    @test issetequal(keys(tensors(ψ, 3).meta[:alias]), [:l, :r, :i, :o])

                    @test tensors(ψ, 1).meta[:alias][:r] === tensors(ψ, 2).meta[:alias][:l]
                    @test tensors(ψ, 2).meta[:alias][:r] === tensors(ψ, 3).meta[:alias][:l]
                    @test tensors(ψ, 3).meta[:alias][:r] === tensors(ψ, 1).meta[:alias][:l]
                end
            end
        end
    end

    @testset "Initialization" begin
        for params in [
            (2, 2, 2, 1),
            (2, 2, 2, 2),
            (4, 4, 4, 16),
            (4, 2, 2, 8),
            (4, 2, 3, 8),
            (6, 2, 2, 4),
            (8, 2, 3, 4),
            # (1, 2, 2, 1),
            # (1, 3, 3, 1),
            # (1, 1, 1, 1),
        ]
            @test rand(MatrixProductOperator{Open}, params...) isa TensorNetwork{MatrixProductOperator{Open}}
        end
    end
end

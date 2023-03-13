@testset "MatrixProductOperator" begin
    using Tenet: TensorNetwork, Operator, Closed, Open, bounds, MatrixProductOperator

    @testset "Types" begin
        @test MatrixProductOperator <: Operator
        @test all(T -> MatrixProductOperator{T} <: Operator, [Open, Closed])
        @test all(T -> MatrixProductOperator{T} <: Operator{T}, [Open, Closed])
        @test all(B -> bounds(MatrixProductOperator{B}) == B, [Open, Closed])
    end

    @testset "Constructor" begin
        @test begin
            arrays = [rand(3, 1, 1), rand(3, 1, 1)]
            MatrixProductOperator(arrays) isa TensorNetwork{MatrixProductOperator{Open}}
        end

        @testset "`Open` boundary" begin
            # product operator
            @test begin
                arrays = [rand(3, 1, 1), rand(3, 3, 1, 1), rand(3, 1, 1)]
                MatrixProductOperator{Open}(arrays) isa TensorNetwork{MatrixProductOperator{Open}}
            end

            @test begin
                arrays = [rand(1, 3, 1), rand(3, 1, 3, 1), rand(3, 1, 1)]
                MatrixProductOperator{Open}(arrays, order = (:l, :i, :r, :o)) isa TensorNetwork{MatrixProductOperator{Open}}
            end
        end

        @testset "`Closed` boundary" begin
            # product state
            @test begin
                arrays = [rand(1, 1, 2, 2), rand(1, 1, 2, 2), rand(1, 1, 2, 2)]
                MatrixProductOperator{Closed}(arrays) isa TensorNetwork{MatrixProductOperator{Closed}}
            end
            
            @test_throws DimensionMismatch begin
                arrays = [rand(3, 1, 1, 3), rand(3, 1, 1, 3), rand(3, 1, 1, 3) ]
                MatrixProductOperator{Closed}(arrays, order = (:l, :i, :o, :r)) isa TensorNetwork{MatrixProductOperator{Closed}}
            end
        end
    end
end
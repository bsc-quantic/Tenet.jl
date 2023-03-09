@testset "MatrixProductOperator" begin
    using Tenet: TensorNetwork, State, Closed, Open, bounds, MatrixProductOperator

    @testset "Types" begin
        @test MatrixProductOperator <: State
        @test all(T -> MatrixProductOperator{T} <: State, [Open, Closed])
        @test all(T -> MatrixProductOperator{T} <: State{T}, [Open, Closed])
        @test all(B -> bounds(MatrixProductOperator{B}) == B, [Open, Closed])
    end

    @testset "Constructor" begin
        # empty constructor

        # @test_skip begin
        #     arrays = [rand(3)]
        #     MatrixProductOperator(arrays)
        # end

        @test_skip begin
            arrays = [rand(1, 3, 1), rand(1, 3, 1)]
            MatrixProductOperator(arrays) isa TensorNetwork{MatrixProductOperator{Open}}
        end

        @testset "`Open` boundary" begin
            # product operator
            @test begin
                arrays = [rand(1, 3, 1), rand(3, 3, 1, 1), rand(3, 1, 1)]
                MatrixProductOperator{Open}(arrays) isa TensorNetwork{MatrixProductOperator{Open}}
            end

            # # entangled state
            # @test begin
            #     arrays = [rand(2, 2), rand(2, 4, 2), rand(4, 1, 2), rand(1, 2)]
            #     MatrixProductState{Open}(arrays) isa TensorNetwork{MatrixProductState{Open}}
            # end

            # alternative constructor
            @test begin
                arrays = [rand(1, 3, 1), rand(3, 3, 1, 1), rand(3, 1, 1)]
                MatrixProductOperator(arrays; bounds = Open) isa TensorNetwork{MatrixProductOperator{Open}}
            end

            # # fail on Open with Closed format
            # @test_throws DimensionMismatch begin
            #     arrays = [rand(1, 1, 2), rand(1, 1, 2), rand(1, 1, 2)]
            #     MatrixProductState{Open}(arrays) isa TensorNetwork{MatrixProductState{Open}}
            # end
        end

        # @testset "`Closed` boundary" begin
        #     # product state
        #     @test begin
        #         arrays = [rand(1, 1, 2), rand(1, 1, 2), rand(1, 1, 2)]
        #         MatrixProductState{Closed}(arrays) isa TensorNetwork{MatrixProductState{Closed}}
        #     end

        #     # entangled state
        #     @test begin
        #         arrays = [rand(3, 4, 2), rand(4, 8, 2), rand(8, 3, 2)]
        #         MatrixProductState{Closed}(arrays) isa TensorNetwork{MatrixProductState{Closed}}
        #     end

        #     # alternative constructor
        #     @test begin
        #         arrays = [rand(1, 1, 2), rand(1, 1, 2), rand(1, 1, 2)]
        #         MatrixProductState(arrays; bounds = Closed) isa TensorNetwork{MatrixProductState{Closed}}
        #     end

        #     # fail on Closed with Open format
        #     @test_throws DimensionMismatch begin
        #         arrays = [rand(1, 2), rand(1, 1, 2), rand(1, 2)]
        #         MatrixProductState{Closed}(arrays) isa TensorNetwork{MatrixProductState{Closed}}
        #     end
        # end
    end
end
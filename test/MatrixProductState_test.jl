@testset "MatrixProductState" begin
    using Tenet: TensorNetwork, State, Closed, Open, bounds, MatrixProductState

    @testset "Types" begin
        @test MatrixProductState <: State
        @test all(T -> MatrixProductState{T} <: State, [Open, Closed])
        @test all(T -> MatrixProductState{T} <: State{T}, [Open, Closed])
        @test all(B -> bounds(MatrixProductState{B}) == B, [Open, Closed])
    end

    @testset "Constructor" begin
        # empty constructor
        # @test_throws BoundsError MatrixProductState([])

        @test_skip begin
            arrays = [rand(2)]
            MatrixProductState(arrays)
        end

        @test_skip begin
            arrays = [rand(1, 2), rand(1, 2)]
            MatrixProductState(arrays) isa TensorNetwork{MatrixProductState{Open}}
        end

        @testset "`Open` boundary" begin
            # product state
            @test begin
                arrays = [rand(1, 2), rand(1, 1, 2), rand(1, 2)]
                MatrixProductState{Open}(arrays) isa TensorNetwork{MatrixProductState{Open}}
            end

            # entangled state
            @test begin
                arrays = [rand(2, 2), rand(2, 4, 2), rand(4, 1, 2), rand(1, 2)]
                MatrixProductState{Open}(arrays) isa TensorNetwork{MatrixProductState{Open}}
            end

            @testset "custom order" begin
                arrays = [rand(3, 1), rand(3, 1, 3), rand(1, 3)]
                ψ = MatrixProductState{Open}(arrays, order = (:r, :p, :l))

                @test ψ isa TensorNetwork{MatrixProductState{Open}}
                @test ψ.meta[:order] == Dict(:r => 1, :p => 2, :l => 3)
            end

            # alternative constructor
            @test begin
                arrays = [rand(1, 2), rand(1, 1, 2), rand(1, 2)]
                MatrixProductState(arrays; bounds = Open) isa TensorNetwork{MatrixProductState{Open}}
            end

            # fail on Open with Closed format
            @test_throws DimensionMismatch begin
                arrays = [rand(1, 1, 2), rand(1, 1, 2), rand(1, 1, 2)]
                MatrixProductState{Open}(arrays) isa TensorNetwork{MatrixProductState{Open}}
            end

            @testset "Metadata" begin
                @testset "alias" begin
                    arrays = [rand(2, 2), rand(2, 2, 2), rand(2, 2)]
                    ψ = MatrixProductState{Open}(arrays, order = (:l, :p, :r))

                    @test ψ.meta[:order] == Dict(:l => 1, :p => 2, :r => 3)

                    @test issetequal(keys(tensors(ψ, 1).meta[:alias]), [:r, :p])
                    @test issetequal(keys(tensors(ψ, 2).meta[:alias]), [:l, :r, :p])
                    @test issetequal(keys(tensors(ψ, 3).meta[:alias]), [:l, :p])

                    @test tensors(ψ, 1).meta[:alias][:r] === tensors(ψ, 2).meta[:alias][:l]
                    @test tensors(ψ, 2).meta[:alias][:r] === tensors(ψ, 3).meta[:alias][:l]
                end
            end
        end

        @testset "`Closed` boundary" begin
            # product state
            @test begin
                arrays = [rand(1, 1, 2), rand(1, 1, 2), rand(1, 1, 2)]
                MatrixProductState{Closed}(arrays) isa TensorNetwork{MatrixProductState{Closed}}
            end

            # entangled state
            @test begin
                arrays = [rand(3, 4, 2), rand(4, 8, 2), rand(8, 3, 2)]
                MatrixProductState{Closed}(arrays) isa TensorNetwork{MatrixProductState{Closed}}
            end

            @testset "custom order" begin
                arrays = [rand(3, 1, 3), rand(3, 1, 3), rand(3, 1, 3)]
                ψ = MatrixProductState{Closed}(arrays, order = (:r, :p, :l))

                @test ψ isa TensorNetwork{MatrixProductState{Closed}}
                @test ψ.meta[:order] == Dict(:r => 1, :p => 2, :l => 3)
            end

            # alternative constructor
            @test begin
                arrays = [rand(1, 1, 2), rand(1, 1, 2), rand(1, 1, 2)]
                MatrixProductState(arrays; bounds = Closed) isa TensorNetwork{MatrixProductState{Closed}}
            end

            # fail on Closed with Open format
            @test_throws DimensionMismatch begin
                arrays = [rand(1, 2), rand(1, 1, 2), rand(1, 2)]
                MatrixProductState{Closed}(arrays) isa TensorNetwork{MatrixProductState{Closed}}
            end

            @testset "Metadata" begin
                @testset "alias" begin
                    arrays = [rand(2, 2, 2), rand(2, 2, 2), rand(2, 2, 2)]
                    ψ = MatrixProductState{Closed}(arrays, order = (:r, :p, :l))

                    @test ψ.meta[:order] == Dict(:r => 1, :p => 2, :l => 3)

                    @test issetequal(keys(tensors(ψ, 1).meta[:alias]), [:l, :r, :p])
                    @test issetequal(keys(tensors(ψ, 2).meta[:alias]), [:l, :r, :p])
                    @test issetequal(keys(tensors(ψ, 3).meta[:alias]), [:l, :r, :p])

                    @test tensors(ψ, 1).meta[:alias][:r] === tensors(ψ, 2).meta[:alias][:l]
                    @test tensors(ψ, 2).meta[:alias][:r] === tensors(ψ, 3).meta[:alias][:l]
                    @test tensors(ψ, 3).meta[:alias][:r] === tensors(ψ, 1).meta[:alias][:l]
                end
            end
        end
    end
end

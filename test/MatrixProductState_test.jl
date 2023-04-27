@testset "MatrixProductState" begin
    using Tenet: TensorNetwork, State, Closed, Open, bounds, MatrixProductState, canonize, conj

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

    @testset "Functions" begin
        using LinearAlgebra: I

        function is_left_orthogonal(A::Tensor{T}) where {T}
            contracted = contract(A, replace(conj(A), labels(A)[2] => :new_ind_name), dims = (labels(A)[1], labels(A)[3]))
            return isapprox(contracted, Matrix{Float64}(I, size(A, 2), size(A, 2)), atol=1e-12)
        end

        function is_right_orthogonal(A::Tensor{T}) where {T}
            contracted = contract(A, replace(conj(A), labels(A)[1] => :new_ind_name), dims = (labels(A)[2], labels(A)[3]))
            return isapprox(contracted, Matrix{Float64}(I, size(A, 1), size(A, 1)), atol=1e-12)
        end

        ψ = rand(MatrixProductState{Open}, 16, 2, 8)

        @testset "chi not limitted" begin
            @testset begin
                ϕ = canonize(ψ, 8)
                @test ϕ isa TensorNetwork{MatrixProductState{Open}}

                A, B = tensors(ϕ, 6), tensors(ϕ, 12)
                @test is_left_orthogonal(A)
                @test is_right_orthogonal(B)
            end

            @testset "limit chi" begin
                ϕ = canonize(ψ, 8; chi = 4)

                A, B = tensors(ϕ, 6), tensors(ϕ, 12)
                @test is_left_orthogonal(A)
                @test is_right_orthogonal(B)
                @test any([any((i != 8 && i != 9 ? size(tensors(ϕ, i)) : (0)) .> 4) for i in 1:length(ϕ)]) == false
            end

            @testset "return singular values" begin
                ϕ, σ = canonize(ψ, 8; return_singular_values = true)

                A, B = tensors(ϕ, 6), tensors(ϕ, 12)
                @test is_left_orthogonal(A)
                @test is_right_orthogonal(B)
                @test length(σ) == 15
            end
        end
    end
end
@testset "MatrixProductState" begin
    using Tenet: TensorNetwork, State, Closed, Open, bounds, MatrixProductState, canonicalize

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

            # custom order
            @test begin
                arrays = [rand(3, 1), rand(3, 1, 3), rand(1, 3)]
                MatrixProductState{Open}(arrays, order = (:r, :p, :l)) isa TensorNetwork{MatrixProductState{Open}}
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

            # custom order
            @test begin
                arrays = [rand(3, 1, 3), rand(3, 1, 3), rand(3, 1, 3)]
                MatrixProductState{Closed}(arrays, order = (:r, :p, :l)) isa TensorNetwork{MatrixProductState{Closed}}
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
        end
    end

    @testset "Functions" begin
        using OMEinsum
        using LinearAlgebra: I

        ψ = rand(MatrixProductState{Open}, 16, 2, 8)

        @testset "canonicalize" begin
            @testset begin
                ϕ = canonicalize(ψ, 8)
                @test canonicalize(ψ, 8) isa TensorNetwork{MatrixProductState{Open}}

                A, B = tensors(ϕ, 6), tensors(ϕ, 12)
                @test isapprox(ein"ijk,ilk->jl"(A, conj(A)), Matrix{Float64}(I, size(A, 2), size(A, 2)))
                @test isapprox(ein"ijk,ljk->il"(B, conj(B)), Matrix{Float64}(I, size(B, 1), size(B, 1)))
            end

            @testset "limit chi" begin
                ϕ = canonicalize(ψ, 8; chi = 4)

                A, B = tensors(ϕ, 6), tensors(ϕ, 12)
                @test isapprox(ein"ijk,ilk->jl"(A, conj(A)), Matrix{Float64}(I, size(A, 2), size(A, 2)))
                @test isapprox(ein"ijk,ljk->il"(B, conj(B)), Matrix{Float64}(I, size(B, 1), size(B, 1)))
                @test any([any((i != 8 && i != 9 ? size(tensors(ϕ, i)) : (0)) .> 4) for i in 1:length(ϕ)]) == false
            end
        end
    end
end
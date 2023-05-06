@testset "MatrixProduct{Operator}" begin
    using Tenet: Operator

    @testset "plug" begin
        @test plug(MatrixProduct{Operator}) === Operator
        @test all(T -> plug(MatrixProduct{Operator,T}) === Operator, [Open, Periodic])
    end

    @testset "boundary" begin
        @test all(B -> boundary(MatrixProduct{Operator,B}) == B, [Open, Periodic])
    end

    @testset "Constructor" begin
        # empty constructor
        @test_throws BoundsError MatrixProduct{Operator}([])

        @test begin
            arrays = [rand(2, 2, 2)]
            MatrixProduct{Operator}(arrays) isa TensorNetwork{MatrixProduct{Operator,Open}}
        end

        @test begin
            arrays = [rand(2, 2, 2), rand(2, 2, 2)]
            MatrixProduct{Operator}(arrays) isa TensorNetwork{MatrixProduct{Operator,Open}}
        end

        @testset "`Open` boundary" begin
            # product operator
            @test begin
                arrays = [rand(1, 2, 2), rand(1, 1, 2, 2), rand(1, 2, 2)]
                MatrixProduct{Operator,Open}(arrays) isa TensorNetwork{MatrixProduct{Operator,Open}}
            end

            # alternative constructor
            @test begin
                arrays = [rand(1, 2, 2), rand(1, 1, 2, 2), rand(1, 2, 2)]
                MatrixProduct{Operator}(arrays; boundary = Open) isa TensorNetwork{MatrixProduct{Operator,Open}}
            end

            # entangling operator
            @test begin
                i = 3
                o = 5
                arrays = [rand(2, i, o), rand(2, 4, i, o), rand(4, i, o)]
                MatrixProduct{Operator,Open}(arrays) isa TensorNetwork{MatrixProduct{Operator,Open}}
            end

            # entangling operator - change order
            @test begin
                i = 3
                o = 5
                arrays = [rand(i, 2, o), rand(2, i, 4, o), rand(4, i, o)]
                MatrixProduct{Operator,Open}(arrays, order = (:l, :i, :r, :o)) isa
                TensorNetwork{MatrixProduct{Operator,Open}}
            end

            # fail on Open with Periodic format
            @test_throws MethodError begin
                arrays = [rand(1, 1, 2, 2), rand(1, 1, 2, 2), rand(1, 1, 2, 2)]
                MatrixProduct{Operator,Open}(arrays) isa TensorNetwork{MatrixProduct{Operator,Open}}
            end

            @testset "metadata" begin
                @testset "alias" begin
                    arrays = [rand(1, 2, 2), rand(1, 1, 2, 2), rand(1, 2, 2)]
                    ψ = MatrixProduct{Operator,Open}(arrays, order = (:l, :r, :i, :o))

                    @test issetequal(keys(tensors(ψ, 1, :out).meta[:alias]), [:r, :i, :o])
                    @test issetequal(keys(tensors(ψ, 2, :out).meta[:alias]), [:l, :r, :i, :o])
                    @test issetequal(keys(tensors(ψ, 3, :out).meta[:alias]), [:l, :i, :o])

                    @test tensors(ψ, 1, :out).meta[:alias][:r] === tensors(ψ, 2, :out).meta[:alias][:l]
                    @test tensors(ψ, 2, :out).meta[:alias][:r] === tensors(ψ, 3, :out).meta[:alias][:l]
                end
            end
        end

        @testset "`Periodic` boundary" begin
            # product operator
            @test begin
                arrays = [rand(1, 1, 2, 2), rand(1, 1, 2, 2), rand(1, 1, 2, 2)]
                MatrixProduct{Operator,Periodic}(arrays) isa TensorNetwork{MatrixProduct{Operator,Periodic}}
            end

            # alternative constructor
            @test begin
                arrays = [rand(1, 1, 2, 2), rand(1, 1, 2, 2), rand(1, 1, 2, 2)]
                MatrixProduct{Operator}(arrays; boundary = Periodic) isa TensorNetwork{MatrixProduct{Operator,Periodic}}
            end

            # entangling operator
            @test begin
                i = 3
                o = 5
                arrays = [rand(2, 4, i, o), rand(4, 8, i, o), rand(8, 2, i, o)]
                MatrixProduct{Operator,Periodic}(arrays) isa TensorNetwork{MatrixProduct{Operator,Periodic}}
            end

            # entangling operator - change order
            @test begin
                i = 3
                o = 5
                arrays = [rand(2, i, 4, o), rand(4, i, 8, o), rand(8, i, 2, o)]
                MatrixProduct{Operator,Periodic}(arrays, order = (:l, :i, :r, :o)) isa
                TensorNetwork{MatrixProduct{Operator,Periodic}}
            end

            # fail on Periodic with Open format
            @test_throws MethodError begin
                arrays = [rand(1, 2, 2), rand(1, 1, 2, 2), rand(1, 2, 2)]
                MatrixProduct{Operator,Periodic}(arrays) isa TensorNetwork{MatrixProduct{Operator,Periodic}}
            end

            @testset "metadata" begin
                @testset "alias" begin
                    arrays = [rand(1, 1, 2, 2), rand(1, 1, 2, 2), rand(1, 1, 2, 2)]
                    ψ = MatrixProduct{Operator,Periodic}(arrays, order = (:l, :r, :i, :o))

                    @test issetequal(keys(tensors(ψ, 1, :out).meta[:alias]), [:l, :r, :i, :o])
                    @test issetequal(keys(tensors(ψ, 2, :out).meta[:alias]), [:l, :r, :i, :o])
                    @test issetequal(keys(tensors(ψ, 3, :out).meta[:alias]), [:l, :r, :i, :o])

                    @test tensors(ψ, 1, :out).meta[:alias][:r] === tensors(ψ, 2, :out).meta[:alias][:l]
                    @test tensors(ψ, 2, :out).meta[:alias][:r] === tensors(ψ, 3, :out).meta[:alias][:l]
                    @test tensors(ψ, 3, :out).meta[:alias][:r] === tensors(ψ, 1, :out).meta[:alias][:l]
                end
            end
        end
    end

    # @testset "Initialization" begin
    #     for params in [
    #         (2, 2, 2, 1),
    #         (2, 2, 2, 2),
    #         (4, 4, 4, 16),
    #         (4, 2, 2, 8),
    #         (4, 2, 3, 8),
    #         (6, 2, 2, 4),
    #         (8, 2, 3, 4),
    #         # (1, 2, 2, 1),
    #         # (1, 3, 3, 1),
    #         # (1, 1, 1, 1),
    #     ]
    #         @test rand(MatrixProduct{Operator,Open}, params...) isa TensorNetwork{MatrixProduct{Operator,Open}}
    #     end
    # end
end

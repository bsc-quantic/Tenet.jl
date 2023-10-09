@testset "MatrixProduct{Operator}" begin
    @testset "plug" begin
        @test plug(MatrixProduct{Operator}) === Operator()
        @test all(T -> plug(MatrixProduct{Operator,T}) === Operator(), [Open, Periodic])
    end

    @testset "boundary" begin
        @test all(B -> boundary(MatrixProduct{Operator,B}) == B(), [Open, Periodic])
    end

    @testset "Constructor" begin
        # empty constructor
        @test_throws Exception MatrixProduct{Operator}([])

        @test begin
            arrays = [rand(2, 2, 2)]
            MatrixProduct{Operator}(arrays) isa MPO{Open}
        end

        @test begin
            arrays = [rand(2, 2, 2), rand(2, 2, 2)]
            MatrixProduct{Operator}(arrays) isa MPO{Open}
        end

        @testset "`Open` boundary" begin
            # product operator
            @test begin
                arrays = [rand(1, 2, 2), rand(1, 1, 2, 2), rand(1, 2, 2)]
                MatrixProduct{Operator,Open}(arrays) isa MPO{Open}
            end

            # alternative constructor
            @test begin
                arrays = [rand(1, 2, 2), rand(1, 1, 2, 2), rand(1, 2, 2)]
                MatrixProduct{Operator}(arrays; boundary = Open) isa MPO{Open}
            end

            # entangling operator
            @test begin
                i = 3
                o = 5
                arrays = [rand(2, i, o), rand(2, 4, i, o), rand(4, i, o)]
                MatrixProduct{Operator,Open}(arrays) isa MPO{Open}
            end

            # entangling operator - change order
            @test begin
                i = 3
                o = 5
                arrays = [rand(i, 2, o), rand(2, i, 4, o), rand(4, i, o)]
                MatrixProduct{Operator,Open}(arrays, order = (:l, :i, :r, :o)) isa MPO{Open}
            end

            # fail on Open with Periodic format
            @test_throws MethodError begin
                arrays = [rand(1, 1, 2, 2), rand(1, 1, 2, 2), rand(1, 1, 2, 2)]
                MatrixProduct{Operator,Open}(arrays) isa MPO{Open}
            end
        end

        @testset "`Periodic` boundary" begin
            # product operator
            @test begin
                arrays = [rand(1, 1, 2, 2), rand(1, 1, 2, 2), rand(1, 1, 2, 2)]
                MatrixProduct{Operator,Periodic}(arrays) isa MPO{Periodic}
            end

            # alternative constructor
            @test begin
                arrays = [rand(1, 1, 2, 2), rand(1, 1, 2, 2), rand(1, 1, 2, 2)]
                MatrixProduct{Operator}(arrays; boundary = Periodic) isa MPO{Periodic}
            end

            # entangling operator
            @test begin
                i = 3
                o = 5
                arrays = [rand(2, 4, i, o), rand(4, 8, i, o), rand(8, 2, i, o)]
                MatrixProduct{Operator,Periodic}(arrays) isa MPO{Periodic}
            end

            # entangling operator - change order
            @test begin
                i = 3
                o = 5
                arrays = [rand(2, i, 4, o), rand(4, i, 8, o), rand(8, i, 2, o)]
                MatrixProduct{Operator,Periodic}(arrays, order = (:l, :i, :r, :o)) isa MPO{Periodic}
            end

            # fail on Periodic with Open format
            @test_throws MethodError begin
                arrays = [rand(1, 2, 2), rand(1, 1, 2, 2), rand(1, 2, 2)]
                MatrixProduct{Operator,Periodic}(arrays) isa MPO{Periodic}
            end
        end

        @testset "`Infinite` boundary" begin
            # product operator
            @test begin
                arrays = [rand(1, 1, 2, 2), rand(1, 1, 2, 2), rand(1, 1, 2, 2)]
                MatrixProduct{Operator,Infinite}(arrays) isa MPO{Infinite}
            end

            # alternative constructor
            @test begin
                arrays = [rand(1, 1, 2, 2), rand(1, 1, 2, 2), rand(1, 1, 2, 2)]
                MatrixProduct{Operator}(arrays; boundary = Infinite) isa MPO{Infinite}
            end

            # entangling operator
            @test begin
                i = 3
                o = 5
                arrays = [rand(2, 4, i, o), rand(4, 8, i, o), rand(8, 2, i, o)]
                MatrixProduct{Operator,Infinite}(arrays) isa MPO{Infinite}
            end

            # entangling operator - change order
            @test begin
                i = 3
                o = 5
                arrays = [rand(2, i, 4, o), rand(4, i, 8, o), rand(8, i, 2, o)]
                MatrixProduct{Operator,Infinite}(arrays, order = (:l, :i, :r, :o)) isa MPO{Infinite}
            end

            # fail on Infinite with Open format
            @test_throws MethodError begin
                arrays = [rand(1, 2, 2), rand(1, 1, 2, 2), rand(1, 2, 2)]
                MatrixProduct{Operator,Infinite}(arrays) isa MPO{Infinite}
            end
        end
    end

    @testset "merge" begin
        @test begin
            arrays = [rand(2, 2), rand(2, 2)]
            mps = MatrixProduct{State,Open}(arrays)
            arrays_o = [rand(2, 2, 2), rand(2, 2, 2)]
            mpo = MatrixProduct{Operator}(arrays_o)
            merge(mps, mpo) isa QuantumTensorNetwork
        end

        @test begin
            arrays = [rand(2, 2), rand(2, 2)]
            mps = MatrixProduct{State,Open}(arrays)
            arrays_o = [rand(2, 2, 2), rand(2, 2, 2)]
            mpo = MatrixProduct{Operator}(arrays_o)
            merge(mpo, mps') isa QuantumTensorNetwork
        end

        @test begin
            arrays = [rand(2, 2, 2), rand(2, 2, 2)]
            mpo = MatrixProduct{Operator}(arrays)
            merge(mpo, mpo') isa QuantumTensorNetwork
        end
    end

    @testset "norm" begin
        mpo = rand(MatrixProduct{Operator,Open}, n = 8, p = 2, χ = 8)
        @test_broken norm(mpo) ≈ 1
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

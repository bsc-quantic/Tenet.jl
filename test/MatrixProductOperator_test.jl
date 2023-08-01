@testset "MatrixProduct{Operator}" begin
    using Tenet: Operator, Composite

    @testset "plug" begin
        @test plug(MatrixProduct{Operator}) === Operator
        @test all(T -> plug(MatrixProduct{Operator,T}) === Operator, [Open, Periodic])
    end

    @testset "boundary" begin
        @test all(B -> boundary(MatrixProduct{Operator,B}) == B, [Open, Periodic])
    end

    @testset "Constructor" begin
        # empty constructor
        @test_throws Exception MatrixProduct{Operator}([])

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

                    # TODO refactor `select` with `tensors` with output selection
                    @test issetequal(keys(only(select(ψ, last(ψ.interlayer)[1])).meta[:alias]), [:r, :i, :o])
                    @test issetequal(keys(only(select(ψ, last(ψ.interlayer)[2])).meta[:alias]), [:l, :r, :i, :o])
                    @test issetequal(keys(only(select(ψ, last(ψ.interlayer)[3])).meta[:alias]), [:l, :i, :o])

                    @test only(select(ψ, last(ψ.interlayer)[1])).meta[:alias][:r] ===
                          only(select(ψ, last(ψ.interlayer)[2])).meta[:alias][:l]
                    @test only(select(ψ, last(ψ.interlayer)[2])).meta[:alias][:r] ===
                          only(select(ψ, last(ψ.interlayer)[3])).meta[:alias][:l]
                end
            end

            @testset "tensors" begin
                arrays = [rand(1, 2, 2), rand(1, 1, 2, 2), rand(1, 2, 2)]
                ψ = MatrixProduct{Operator,Open}(arrays, order = (:l, :r, :i, :o))

                @test tensors(ψ, 1) isa Tensor
                @test size(tensors(ψ)) |> first == length(ψ) == 3
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

                    # TODO refactor `select` with `tensors` with output selection
                    @test issetequal(keys(only(select(ψ, first(ψ.interlayer)[1])).meta[:alias]), [:l, :r, :i, :o])
                    @test issetequal(keys(only(select(ψ, first(ψ.interlayer)[2])).meta[:alias]), [:l, :r, :i, :o])
                    @test issetequal(keys(only(select(ψ, first(ψ.interlayer)[3])).meta[:alias]), [:l, :r, :i, :o])

                    @test only(select(ψ, first(ψ.interlayer)[1])).meta[:alias][:r] ===
                          only(select(ψ, first(ψ.interlayer)[2])).meta[:alias][:l]
                    @test only(select(ψ, first(ψ.interlayer)[2])).meta[:alias][:r] ===
                          only(select(ψ, first(ψ.interlayer)[3])).meta[:alias][:l]
                    @test only(select(ψ, first(ψ.interlayer)[3])).meta[:alias][:r] ===
                          only(select(ψ, first(ψ.interlayer)[1])).meta[:alias][:l]
                end

                @testset "tensors" begin
                    arrays = [rand(1, 1, 2, 2), rand(1, 1, 2, 2), rand(1, 1, 2, 2)]
                    ψ = MatrixProduct{Operator,Periodic}(arrays, order = (:l, :r, :i, :o))

                    @test tensors(ψ, 1) isa Tensor
                    @test size(tensors(ψ)) |> first == length(ψ) == 3
                end
            end
        end

        @testset "`Infinite` boundary" begin
            # product operator
            @test begin
                arrays = [rand(1, 1, 2, 2), rand(1, 1, 2, 2), rand(1, 1, 2, 2)]
                MatrixProduct{Operator,Infinite}(arrays) isa TensorNetwork{MatrixProduct{Operator,Infinite}}
            end

            # alternative constructor
            @test begin
                arrays = [rand(1, 1, 2, 2), rand(1, 1, 2, 2), rand(1, 1, 2, 2)]
                MatrixProduct{Operator}(arrays; boundary = Infinite) isa TensorNetwork{MatrixProduct{Operator,Infinite}}
            end

            # entangling operator
            @test begin
                i = 3
                o = 5
                arrays = [rand(2, 4, i, o), rand(4, 8, i, o), rand(8, 2, i, o)]
                MatrixProduct{Operator,Infinite}(arrays) isa TensorNetwork{MatrixProduct{Operator,Infinite}}
            end

            # entangling operator - change order
            @test begin
                i = 3
                o = 5
                arrays = [rand(2, i, 4, o), rand(4, i, 8, o), rand(8, i, 2, o)]
                MatrixProduct{Operator,Infinite}(arrays, order = (:l, :i, :r, :o)) isa
                TensorNetwork{MatrixProduct{Operator,Infinite}}
            end

            # fail on Infinite with Open format
            @test_throws MethodError begin
                arrays = [rand(1, 2, 2), rand(1, 1, 2, 2), rand(1, 2, 2)]
                MatrixProduct{Operator,Infinite}(arrays) isa TensorNetwork{MatrixProduct{Operator,Infinite}}
            end

            @testset "tensors" begin
                arrays = [rand(1, 1, 2, 2), rand(1, 1, 2, 2), rand(1, 1, 2, 2)]
                ψ = MatrixProduct{Operator,Infinite}(arrays, order = (:l, :r, :i, :o))

                @test_throws ArgumentError tensors(ψ)
                @test tensors(ψ, 1) == tensors(ψ, 4)
                @test tensors(ψ, -1) == tensors(ψ, 3)
                @test length(ψ) == Inf
            end

            @testset "metadata" begin
                @testset "alias" begin
                    arrays = [rand(1, 1, 2, 2), rand(1, 1, 2, 2), rand(1, 1, 2, 2)]
                    ψ = MatrixProduct{Operator,Infinite}(arrays, order = (:l, :r, :i, :o))

                    # TODO refactor `select` with `tensors` with output selection
                    @test issetequal(keys(only(select(ψ, first(ψ.interlayer)[1])).meta[:alias]), [:l, :r, :i, :o])
                    @test issetequal(keys(only(select(ψ, first(ψ.interlayer)[2])).meta[:alias]), [:l, :r, :i, :o])
                    @test issetequal(keys(only(select(ψ, first(ψ.interlayer)[3])).meta[:alias]), [:l, :r, :i, :o])

                    @test only(select(ψ, first(ψ.interlayer)[1])).meta[:alias][:r] ===
                          only(select(ψ, first(ψ.interlayer)[2])).meta[:alias][:l]
                    @test only(select(ψ, first(ψ.interlayer)[2])).meta[:alias][:r] ===
                          only(select(ψ, first(ψ.interlayer)[3])).meta[:alias][:l]
                    @test only(select(ψ, first(ψ.interlayer)[3])).meta[:alias][:r] ===
                          only(select(ψ, first(ψ.interlayer)[1])).meta[:alias][:l]
                end
            end
        end
    end

    @testset "hcat" begin
        @test begin
            arrays = [rand(2, 2), rand(2, 2)]
            mps = MatrixProduct{State,Open}(arrays)
            arrays_o = [rand(2, 2, 2), rand(2, 2, 2)]
            mpo = MatrixProduct{Operator}(arrays_o)
            hcat(mps, mpo) isa TensorNetwork{<:Composite}
        end

        @test begin
            arrays = [rand(2, 2), rand(2, 2)]
            mps = MatrixProduct{State,Open}(arrays)
            arrays_o = [rand(2, 2, 2), rand(2, 2, 2)]
            mpo = MatrixProduct{Operator}(arrays_o)
            hcat(mpo, mps) isa TensorNetwork{<:Composite}
        end

        @test begin
            arrays = [rand(2, 2, 2), rand(2, 2, 2)]
            mpo = MatrixProduct{Operator}(arrays)
            hcat(mpo, mpo) isa TensorNetwork{<:Composite}
        end
    end

    @testset "norm" begin
        mpo = rand(MatrixProduct{Operator,Open}, n = 8, p = 2, χ = 8)
        @test norm(mpo) ≈ 1
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

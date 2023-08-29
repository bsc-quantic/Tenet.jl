@testset "MatrixProduct{State}" begin
    using Tenet: Composite

    @testset "plug" begin
        @test plug(MatrixProduct{State}) === State
        @test all(T -> plug(MatrixProduct{State,T}) === State, [Open, Periodic])
    end

    @testset "boundary" begin
        @test all(B -> boundary(MatrixProduct{State,B}) == B, [Open, Periodic])
    end

    @testset "Constructor" begin
        # empty constructor
        @test_throws Exception MatrixProduct{State}([])

        @test begin
            arrays = [rand(1, 2)]
            MatrixProduct{State}(arrays) isa TensorNetwork{MatrixProduct{State,Open}}
        end

        @test begin
            arrays = [rand(1, 2), rand(1, 2)]
            MatrixProduct{State}(arrays) isa TensorNetwork{MatrixProduct{State,Open}}
        end

        @testset "`Open` boundary" begin
            # product state
            @test begin
                arrays = [rand(1, 2), rand(1, 1, 2), rand(1, 2)]
                MatrixProduct{State,Open}(arrays) isa TensorNetwork{MatrixProduct{State,Open}}
            end

            # entangled state
            @test begin
                arrays = [rand(2, 2), rand(2, 4, 2), rand(4, 1, 2), rand(1, 2)]
                MatrixProduct{State,Open}(arrays) isa TensorNetwork{MatrixProduct{State,Open}}
            end

            @testset "custom order" begin
                arrays = [rand(3, 1), rand(3, 1, 3), rand(1, 3)]
                ψ = MatrixProduct{State,Open}(arrays, order = (:r, :o, :l))

                @test ψ isa TensorNetwork{MatrixProduct{State,Open}}
            end

            # alternative constructor
            @test begin
                arrays = [rand(1, 2), rand(1, 1, 2), rand(1, 2)]
                MatrixProduct{State}(arrays; boundary = Open) isa TensorNetwork{MatrixProduct{State,Open}}
            end

            # fail on Open with Periodic format
            @test_throws Exception begin
                arrays = [rand(1, 1, 2), rand(1, 1, 2), rand(1, 1, 2)]
                MatrixProduct{State,Open}(arrays) isa TensorNetwork{MatrixProduct{State,Open}}
            end

            @testset "rand" begin
                # 4 => χ < maximum possible χ for the given parameters
                # 32 => χ > maximum possible χ for the given parameters
                @testset "χ = $χ" for χ in [4, 32]
                    ψ = rand(MatrixProduct{State,Open}, n = 7, p = 2, χ = χ)

                    @test ψ isa TensorNetwork{MatrixProduct{State,Open}}
                    @test length(ψ) == 7
                    @test maximum(vind -> size(ψ, vind), inds(ψ, :inner)) <= 32
                end
            end

            @testset "metadata" begin
                @testset "alias" begin
                    arrays = [rand(2, 2), rand(2, 2, 2), rand(2, 2)]
                    ψ = MatrixProduct{State,Open}(arrays, order = (:l, :o, :r))

                    @test issetequal(keys(tensors(ψ, 1).meta[:alias]), [:r, :o])
                    @test issetequal(keys(tensors(ψ, 2).meta[:alias]), [:l, :r, :o])
                    @test issetequal(keys(tensors(ψ, 3).meta[:alias]), [:l, :o])

                    @test tensors(ψ, 1).meta[:alias][:r] === tensors(ψ, 2).meta[:alias][:l]
                    @test tensors(ψ, 2).meta[:alias][:r] === tensors(ψ, 3).meta[:alias][:l]
                end
            end
        end

        @testset "`Periodic` boundary" begin
            # product state
            @test begin
                arrays = [rand(1, 1, 2), rand(1, 1, 2), rand(1, 1, 2)]
                MatrixProduct{State,Periodic}(arrays) isa TensorNetwork{MatrixProduct{State,Periodic}}
            end

            # entangled state
            @test begin
                arrays = [rand(3, 4, 2), rand(4, 8, 2), rand(8, 3, 2)]
                MatrixProduct{State,Periodic}(arrays) isa TensorNetwork{MatrixProduct{State,Periodic}}
            end

            @testset "custom order" begin
                arrays = [rand(3, 1, 3), rand(3, 1, 3), rand(3, 1, 3)]
                ψ = MatrixProduct{State,Periodic}(arrays, order = (:r, :o, :l))

                @test ψ isa TensorNetwork{MatrixProduct{State,Periodic}}
            end

            # alternative constructor
            @test begin
                arrays = [rand(1, 1, 2), rand(1, 1, 2), rand(1, 1, 2)]
                MatrixProduct{State}(arrays; boundary = Periodic) isa TensorNetwork{MatrixProduct{State,Periodic}}
            end

            # fail on Periodic with Open format
            @test_throws Exception begin
                arrays = [rand(1, 2), rand(1, 1, 2), rand(1, 2)]
                MatrixProduct{State,Periodic}(arrays) isa TensorNetwork{MatrixProduct{State,Periodic}}
            end

            @testset "metadata" begin
                @testset "alias" begin
                    arrays = [rand(2, 2, 2), rand(2, 2, 2), rand(2, 2, 2)]
                    ψ = MatrixProduct{State,Periodic}(arrays, order = (:r, :o, :l))

                    @test issetequal(keys(tensors(ψ, 1).meta[:alias]), [:l, :r, :o])
                    @test issetequal(keys(tensors(ψ, 2).meta[:alias]), [:l, :r, :o])
                    @test issetequal(keys(tensors(ψ, 3).meta[:alias]), [:l, :r, :o])

                    @test tensors(ψ, 1).meta[:alias][:r] === tensors(ψ, 2).meta[:alias][:l]
                    @test tensors(ψ, 2).meta[:alias][:r] === tensors(ψ, 3).meta[:alias][:l]
                    @test tensors(ψ, 3).meta[:alias][:r] === tensors(ψ, 1).meta[:alias][:l]
                end
            end
        end

        @testset "`Infinite` boundary" begin
            # product state
            @test begin
                arrays = [rand(1, 1, 2), rand(1, 1, 2), rand(1, 1, 2)]
                MatrixProduct{State,Infinite}(arrays) isa TensorNetwork{MatrixProduct{State,Infinite}}
            end

            # entangled state
            @test begin
                arrays = [rand(3, 4, 2), rand(4, 8, 2), rand(8, 3, 2)]
                MatrixProduct{State,Infinite}(arrays) isa TensorNetwork{MatrixProduct{State,Infinite}}
            end

            @testset "custom order" begin
                arrays = [rand(3, 1, 3), rand(3, 1, 3), rand(3, 1, 3)]
                ψ = MatrixProduct{State,Infinite}(arrays, order = (:r, :o, :l))

                @test ψ isa TensorNetwork{MatrixProduct{State,Infinite}}
            end

            # alternative constructor
            @test begin
                arrays = [rand(1, 1, 2), rand(1, 1, 2), rand(1, 1, 2)]
                MatrixProduct{State}(arrays; boundary = Infinite) isa TensorNetwork{MatrixProduct{State,Infinite}}
            end

            # fail on Infinite with Open format
            @test_throws Exception begin
                arrays = [rand(1, 2), rand(1, 1, 2), rand(1, 2)]
                MatrixProduct{State,Infinite}(arrays) isa TensorNetwork{MatrixProduct{State,Infinite}}
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

                @testset "tensors" begin
                    arrays = [rand(1, 1, 2), rand(1, 1, 2), rand(1, 1, 2)]
                    ψ = MatrixProduct{State,Infinite}(arrays, order = (:l, :r, :o))

                    @test tensors(ψ, 1) isa Tensor
                    @test length(ψ) == Inf
                    @test tensors(ψ, 4) == tensors(ψ, 1)
                    @test tensors(ψ, 0) == tensors(ψ, 3)
                end
            end
        end
    end

    @testset "hcat" begin
        @test begin
            arrays = [rand(2, 2), rand(2, 2)]
            mps = MatrixProduct{State,Open}(arrays)
            hcat(mps, mps) isa TensorNetwork{<:Composite}
        end

        @test begin
            arrays = [rand(1, 1, 2), rand(1, 1, 2)]
            mps = MatrixProduct{State,Periodic}(arrays)
            hcat(mps, mps) isa TensorNetwork{<:Composite}
        end
    end

    @testset "norm" begin
        mps = rand(MatrixProduct{State,Open}, n = 8, p = 2, χ = 8)
        @test norm(mps) ≈ 1
    end
end

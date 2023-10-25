@testset "MatrixProduct{State}" begin
    @testset "plug" begin
        @test plug(MatrixProduct{State}) == State()
        @test all(T -> plug(MatrixProduct{State,T}) == State(), [Open, Periodic])
    end

    @testset "boundary" begin
        @test all(B -> boundary(MatrixProduct{State,B}) == B(), [Open, Periodic])
    end

    @testset "Constructor" begin
        # empty constructor
        @test_throws Exception MatrixProduct{State}([])

        @test begin
            arrays = [rand(1, 2)]
            MatrixProduct{State}(arrays) isa QuantumTensorNetwork
        end

        @test begin
            arrays = [rand(1, 2), rand(1, 2)]
            MatrixProduct{State}(arrays) isa QuantumTensorNetwork
        end

        @testset "`Open` boundary" begin
            # product state
            @test begin
                arrays = [rand(1, 2), rand(1, 1, 2), rand(1, 2)]
                MatrixProduct{State,Open}(arrays) isa QuantumTensorNetwork
            end

            # entangled state
            @test begin
                arrays = [rand(2, 2), rand(2, 4, 2), rand(4, 1, 2), rand(1, 2)]
                MatrixProduct{State,Open}(arrays) isa QuantumTensorNetwork
            end

            @testset "custom order" begin
                arrays = [rand(3, 1), rand(3, 1, 3), rand(1, 3)]
                ψ = MatrixProduct{State,Open}(arrays, order = (:r, :o, :l))

                @test ψ isa QuantumTensorNetwork
            end

            # alternative constructor
            @test begin
                arrays = [rand(1, 2), rand(1, 1, 2), rand(1, 2)]
                MatrixProduct{State}(arrays; boundary = Open) isa QuantumTensorNetwork
            end

            # fail on Open with Periodic format
            @test_throws Exception begin
                arrays = [rand(1, 1, 2), rand(1, 1, 2), rand(1, 1, 2)]
                MatrixProduct{State,Open}(arrays) isa QuantumTensorNetwork
            end

            @testset "rand" begin
                # 4 => χ < maximum possible χ for the given parameters
                # 32 => χ > maximum possible χ for the given parameters
                @testset "χ = $χ" for χ in [4, 32]
                    ψ = rand(MatrixProduct{State,Open}, n = 7, p = 2, χ = χ)

                    @test ψ isa QuantumTensorNetwork
                    @test length(tensors(ψ)) == 7
                    @test maximum(vind -> size(ψ, vind), inds(ψ, :inner)) <= 32
                end
            end
        end

        @testset "`Periodic` boundary" begin
            # product state
            @test begin
                arrays = [rand(1, 1, 2), rand(1, 1, 2), rand(1, 1, 2)]
                MatrixProduct{State,Periodic}(arrays) isa QuantumTensorNetwork
            end

            # entangled state
            @test begin
                arrays = [rand(3, 4, 2), rand(4, 8, 2), rand(8, 3, 2)]
                MatrixProduct{State,Periodic}(arrays) isa QuantumTensorNetwork
            end

            @testset "custom order" begin
                arrays = [rand(3, 1, 3), rand(3, 1, 3), rand(3, 1, 3)]
                ψ = MatrixProduct{State,Periodic}(arrays, order = (:r, :o, :l))

                @test ψ isa QuantumTensorNetwork
            end

            # alternative constructor
            @test begin
                arrays = [rand(1, 1, 2), rand(1, 1, 2), rand(1, 1, 2)]
                MatrixProduct{State}(arrays; boundary = Periodic) isa QuantumTensorNetwork
            end

            # fail on Periodic with Open format
            @test_throws Exception begin
                arrays = [rand(1, 2), rand(1, 1, 2), rand(1, 2)]
                MatrixProduct{State,Periodic}(arrays) isa QuantumTensorNetwork
            end
        end

        # @testset "`Infinite` boundary" begin
        #     # product state
        #     @test skip = true begin
        #         arrays = [rand(1, 1, 2), rand(1, 1, 2), rand(1, 1, 2)]
        #         MatrixProduct{State,Infinite}(arrays) isa MPS{Infinite}
        #     end

        #     # entangled state
        #     @test skip = true begin
        #         arrays = [rand(3, 4, 2), rand(4, 8, 2), rand(8, 3, 2)]
        #         MatrixProduct{State,Infinite}(arrays) isa MPS{Infinite}
        #     end

        #     @testset "custom order" begin
        #         arrays = [rand(3, 1, 3), rand(3, 1, 3), rand(3, 1, 3)]
        #         ψ = MatrixProduct{State,Infinite}(arrays, order = (:r, :o, :l))

        #         @test skip = true ψ isa MPS{Infinite}
        #     end

        #     # alternative constructor
        #     @test skip = true begin
        #         arrays = [rand(1, 1, 2), rand(1, 1, 2), rand(1, 1, 2)]
        #         MatrixProduct{State}(arrays; boundary = Infinite) isa MPS{Infinite}
        #     end

        #     # fail on Infinite with Open format
        #     @test_throws skip = true Exception begin
        #         arrays = [rand(1, 2), rand(1, 1, 2), rand(1, 2)]
        #         MatrixProduct{State,Infinite}(arrays) isa MPS{Infinite}
        #     end

        #     # @testset "tensors" begin
        #     #     arrays = [rand(1, 1, 2), rand(1, 1, 2), rand(1, 1, 2)]
        #     #     ψ = MatrixProduct{State,Infinite}(arrays, order = (:l, :r, :o))

        #     #     @test tensors(ψ, 1) isa Tensor
        #     #     @test tensors(ψ, 4) == tensors(ψ, 1)
        #     #     @test tensors(ψ, 0) == tensors(ψ, 3)
        #     # end
        # end
    end

    @testset "merge" begin
        @test begin
            arrays = [rand(2, 2), rand(2, 2)]
            mps = MatrixProduct{State,Open}(arrays)
            merge(mps, mps') isa QuantumTensorNetwork
        end

        @test begin
            arrays = [rand(1, 1, 2), rand(1, 1, 2)]
            mps = MatrixProduct{State,Periodic}(arrays)
            merge(mps, mps') isa QuantumTensorNetwork
        end
    end

    @testset "norm" begin
        mps = rand(MatrixProduct{State,Open}, n = 8, p = 2, χ = 8)
        @test norm(mps) ≈ 1
    end
end

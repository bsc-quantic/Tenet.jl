@testset "Transformations" begin
    using Tenet: transform, transform!, Transformation

    @testset "transform" begin
        struct MockTransformation <: Transformation end
        Tenet.transform!(::TensorNetwork, ::MockTransformation) = nothing

        tn = rand(TensorNetwork, 10, 3)
        @test isnothing(transform!(tn, MockTransformation()))
        @test isnothing(transform!(tn, MockTransformation))
        @test transform!(tn, [MockTransformation]) === tn
        @test transform!(tn, [MockTransformation()]) === tn

        @test transform(tn, MockTransformation) isa TensorNetwork
        @test transform(tn, MockTransformation()) isa TensorNetwork
        @test transform(tn, [MockTransformation]) isa TensorNetwork
        @test transform(tn, [MockTransformation()]) isa TensorNetwork
    end

    @testset "HyperindConverter" begin
        using Tenet: HyperindConverter
        using DeltaArrays: DeltaArray

        t_ij = Tensor(zeros(2, 2), (:i, :j))
        t_ik = Tensor(zeros(2, 2), (:i, :k))
        t_ilm = Tensor(zeros(2, 2, 2), (:i, :l, :m))
        t_lm = Tensor(zeros(2, 2), (:l, :m))
        tn = TensorNetwork([t_ij, t_ik, t_ilm, t_lm])

        transform!(tn, HyperindConverter)
        @test isempty(hyperinds(tn))
        @test any(t -> get(t.meta, :dual, nothing) == :i && parent(t) isa DeltaArray, tensors(tn))

        # TODO @test issetequal(neighbours())
    end

    @testset "Local transformations" begin
        @testset "DiagonalReduction" begin
            using Tenet: DiagonalReduction, find_diag_axes

            data = zeros(Float64, 2, 2, 2, 2, 2)
            data2 = zeros(Float64, 2, 2, 2)
            for i in 1:2
                for j in 1:2
                    for k in 1:2
                        # In data the 1st-4th and 2nd-5th indices are diagonal
                        data[i, j, k, i, j] = k
                        data[j, i, k, j, i] = k + 2
                    end

                    data2[i, i, i] = 1 # all indices are diagonal in data2
                end
            end

            A = Tensor(data, (:i, :j, :k, :l, :m))
            B = Tensor(data2, (:j, :n, :o))
            C = Tensor(rand(2, 2, 2), (:k, :p, :q))

            @test issetequal(find_diag_axes(parent(A)), [(1, 4), (2, 5)])
            @test issetequal(find_diag_axes(parent(B)), [(1, 2), (1, 3), (2, 3)])

            tn = TensorNetwork([A, B, C])
            reduced = transform(tn, DiagonalReduction)

            # Test that all tensors in reduced have no diagonals
            for tensor in reduced.tensors
                @test isempty(find_diag_axes(parent(tensor)))
            end

            # Test that the resulting contraction contains the same as the original
            @test contract(reduced) |> parent |> sum ≈ contract(tn) |> parent |> sum
        end

        @testset "RankSimplification" begin
            using Tenet: RankSimplification

            # create a tensor network where tensors B and D can be absorbed
            A = Tensor(rand(2, 2, 2, 2), (:i, :j, :k, :l))
            B = Tensor(rand(2, 2), (:i, :m))
            C = Tensor(rand(2, 2, 2), (:m, :n, :o))
            D = Tensor(rand(2,), (:p,))
            E = Tensor(rand(2, 2, 2, 2), (:o, :p, :q, :j))

            tn = TensorNetwork([A, B, C, D, E])
            reduced = transform(tn, RankSimplification)

            # Test that the resulting tn contains no tensors with larger rank than the original
            rank = length ∘ size ∘ parent
            @test max(rank(tensors(reduced)) ≤ max(rank(tensors(tn))))

            # Test that the resulting tn contains <= tensors than the original
            @test length(tensors(reduced)) ≤ length(tensors(tn))

            # Test that the resulting contraction contains the same as the original
            @test contract(reduced) ≈ contract(tn)
        end
    end
end

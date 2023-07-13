@testset "Transformations" begin
    using Tenet: transform, transform!, Transformation
    using DeltaArrays: DeltaArray

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

        t_ij = Tensor(zeros(2, 2), (:i, :j))
        t_ik = Tensor(zeros(2, 2), (:i, :k))
        t_ilm = Tensor(zeros(2, 2, 2), (:i, :l, :m))
        t_lm = Tensor(zeros(2, 2), (:l, :m))
        tn = TensorNetwork([t_ij, t_ik, t_ilm, t_lm])

        transform!(tn, HyperindConverter)
        @test isempty(labels(tn, :hyper))
        @test any(t -> get(t.meta, :dual, nothing) == :i && parent(t) isa DeltaArray, tensors(tn))

        # TODO @test issetequal(neighbours())
    end

    @testset "DiagonalReduction" begin
        using Tenet: DiagonalReduction, find_diag_axes

        function has_diagonal_in_innerinds(tensor, innerinds)
            for (i, j) in find_diag_axes(parent(tensor))
                idx_i, idx_j = labels(tensor)[i], labels(tensor)[j]

                if idx_i ∈ innerinds || idx_j ∈ innerinds
                    return true
                end
            end
            return false
        end

        @testset "innerinds" begin
            using Tensors: parenttype

            data = zeros(Float64, 2, 2, 2, 2)
            for i in 1:2
                for j in 1:2
                    for k in 1:2
                        # In data the 1st-2th are diagonal
                        data[i, i, j, k] = k
                    end
                end
            end

            A = Tensor(data, (:i, :j, :k, :l))
            B = Tensor(rand(2, 2), (:i, :m))
            C = Tensor(rand(2, 2), (:j, :n))

            @test issetequal(find_diag_axes(A), [[:i, :j]])

            tn = TensorNetwork([A, B, C])
            reduced = transform(tn, DiagonalReduction)

            @test all(
                isempty ∘ find_diag_axes,
                filter(tensor -> !(parenttype(typeof(tensor)) <: DeltaArray), tensors(reduced)),
            )

            # Test that the resulting contraction returns the same as the original
            # @test contract(reduced) ≈ contract(tn)
        end

        @testset "openinds" begin
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

            @test issetequal(find_diag_axes(A), [[:i, :l], [:j, :m]])
            @test issetequal(find_diag_axes(B), [[:j, :n, :o]])

            tn = TensorNetwork([A, B, C])
            reduced = transform(tn, DiagonalReduction)

            # Test that all tensors (that are no COPY tensors) in reduced have no
            #  diagonals in the that are in innerinds
            for tensor in filter(t -> !(parent(t) isa DeltaArray), tensors(reduced))
                @test has_diagonal_in_innerinds(tensor, labels(reduced, set = :inner)) == false
            end

            # Test that the resulting contraction returns the same as the original
            # @test contract(reduced) ≈ contract(tn)
        end
    end

    @testset "RankSimplification" begin
        using Tenet: RankSimplification

        # create a tensor network where tensors B and D can be absorbed
        A = Tensor(rand(2, 2, 2, 2), (:i, :j, :k, :l))
        B = Tensor(rand(2, 2), (:i, :m))
        C = Tensor(rand(2, 2, 2), (:m, :n, :o))
        D = Tensor(rand(2), (:p,))
        E = Tensor(rand(2, 2, 2, 2), (:o, :p, :q, :j))

        tn = TensorNetwork([A, B, C, D, E])
        reduced = transform(tn, RankSimplification)

        # Test that the resulting tn contains no tensors with larger rank than the original
        rank = length ∘ size ∘ parent
        @test max(rank(tensors(reduced)) ≤ max(rank(tensors(tn))))

        # Test that the resulting tn contains <= tensors than the original
        @test length(tensors(reduced)) ≤ length(tensors(tn))

        # Test that the resulting contraction contains the same as the original
        # TODO: the permutation will not be necessary if https://github.com/bsc-quantic/Tensors.jl/issues/27 is fixed
        contracted_reduced = contract(reduced)
        contracted_tn = contract(tn)

        # Calculate the permutation for the `reduced` tensor labels to match `tn`
        perm = sortperm(collect(labels(contracted_reduced)), by = x -> findfirst(==(x), collect(labels(contracted_tn))))
        @test permutedims(contracted_reduced, perm) ≈ contracted_tn
    end

    @testset "AntiDiagonalGauging" begin
        using Tenet: AntiDiagonalGauging, find_anti_diag_axes, labels

        function has_antidiagonal_in_innerinds(tensor, innerinds)
            for (i, j) in find_anti_diag_axes(parent(tensor))
                idx_i, idx_j = labels(tensor)[i], labels(tensor)[j]

                if idx_i ∈ innerinds || idx_j ∈ innerinds
                    return true
                end
            end
            return false
        end

        d = 2  # size of indices

        data = zeros(Float64, d, d, d, d, d)
        data2 = zeros(Float64, d, d, d)
        for i in 1:d
            for j in 1:d
                for k in 1:d
                    # In data_anti the 1st-4th and 2nd-5th indices are antidiagonal
                    data[i, j, k, d-i+1, d-j+1] = k
                    data[j, i, k, d-j+1, d-i+1] = k + 2

                    data2[i, d-i+1, k] = 1  # 1st-2nd indices are antidiagonal in data2_anti
                end
            end
        end

        A = Tensor(data, (:i, :j, :k, :l, :m))
        B = Tensor(data2, (:j, :n, :o))
        C = Tensor(rand(d, d, d), (:k, :p, :q))

        @test issetequal(find_anti_diag_axes(parent(A)), [(1, 4), (2, 5)])
        @test issetequal(find_anti_diag_axes(parent(B)), [(1, 2)])

        tn = TensorNetwork([A, B, C])
        gauged = transform(tn, AntiDiagonalGauging)

        # Test that all tensors in gauged have no antidiagonals
        for tensor in tensors(gauged)
            @test has_antidiagonal_in_innerinds(tensor, labels(gauged, set = :inner)) == false
        end

        # Test that the resulting contraction is the same as the original
        # TODO: Change for: @test contract(gauged) ≈ contract(tn), when is fixed
        A_2, B_2, C_2 = tensors(gauged)
        @test contract(A, contract(B, C)) ≈ contract(A_2, contract(B_2, C_2))
    end

    @testset "ColumnReduction" begin
        using Tenet: ColumnReduction, find_zero_columns, labels

        @testset "rank reduction" begin
            data = rand(3, 3, 3)
            data[:, 1:2, :] .= 0 # 1st and 2nd column of the 2nd dimension are zero
            # Since there is only one non-zero column, the whole 2nd dimension can be reduced

            A = Tensor(data, (:i, :j, :k))
            B = Tensor(rand(3, 3), (:j, :l))
            C = Tensor(rand(3, 3), (:j, :m))

            @test issetequal(find_zero_columns(parent(A)), [(2, 1), (2, 2)])

            tn = TensorNetwork([A, B, C])
            reduced = transform(tn, ColumnReduction)

            # Test that all the tensors in reduced have no columns and they do not have the 2nd :j index
            for tensor in tensors(reduced)
                @test isempty(find_zero_columns(parent(tensor)))
                @test :j ∉ labels(tensor)
            end

            @test length(tn.indices) > length(reduced.indices)

            # Test that the resulting contraction is the same as the original
            # TODO: Change for: @test contract(reduced) ≈ contract(tn), when is fixed
            A_2, B_2, C_2 = tensors(reduced)
            @test contract(A, contract(B, C, dims = [])) ≈ contract(A_2, contract(B_2, C_2, dims = []))
        end

        @testset "index size reduction" begin
            data = rand(3, 3, 3)
            data[:, 2, :] .= 0 # 2nd column of the 2nd dimension can be reduced

            A = Tensor(data, (:i, :j, :k))
            B = Tensor(rand(3, 3), (:j, :l))
            C = Tensor(rand(3, 3), (:j, :m))

            @test issetequal(find_zero_columns(parent(A)), [(2, 2)])

            tn = TensorNetwork([A, B, C])
            reduced = transform(tn, ColumnReduction)

            # Test that all the tensors in reduced have no columns and they have smaller dimensions in the 2nd :j index
            for tensor in tensors(reduced)
                @test isempty(Tenet.find_zero_columns(parent(tensor)))
                # @assert size(tensor, :j) == 2
            end

            @test length(tn.indices) == length(reduced.indices)

            # Test that the resulting contraction is the same as the original
            # TODO: Change for: @test contract(reduced) ≈ contract(tn), when is fixed
            A_2, B_2, C_2 = tensors(reduced)
            @test contract(A, contract(B, C, dims = [])) ≈ contract(A_2, contract(B_2, C_2, dims = []))
        end
    end

    @testset "SplitSimplification" begin
        using Tenet: SplitSimplification

        v1 = Tensor([1, 2, 3], (:i,))
        v2 = Tensor([4, 5, 6], (:j,))
        m1 = Tensor(rand(3, 3), (:k, :l))

        t1 = contract(v1, v2)
        tensor = contract(t1, m1) # Define a tensor which can be splitted in three

        tn = TensorNetwork([tensor, Tensor(rand(3, 3, 3), (:k, :m, :n)), Tensor(rand(3, 3, 3), (:l, :n, :o))])
        reduced = transform(tn, SplitSimplification)

        # Test that the new tensors in reduced are smaller than the deleted ones
        deleted_tensors = filter(t -> labels(t) ∉ labels.(tensors(reduced)), tensors(tn))
        new_tensors = filter(t -> labels(t) ∉ labels.(tensors(tn)), tensors(reduced))

        smallest_deleted = minimum(prod ∘ size, deleted_tensors)
        largest_new = maximum(prod ∘ size, new_tensors)

        @test smallest_deleted > largest_new

        # Test that the resulting contraction is the same as the original
        # TODO: Change for: @test contract(reduced) ≈ contract(tn), when is fixed
        A_2, B_2, C_2, D_2, E_2 = tensors(reduced)
        c_reduced = contract(contract(contract(contract(A_2, B_2), C_2), D_2), E_2)
        @test contract(contract(tensors(tn)[1], tensors(tn)[2]), tensors(tn)[3]) ≈ c_reduced
    end
end

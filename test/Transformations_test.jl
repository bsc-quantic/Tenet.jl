@testset "Transformations" begin
    using Tenet: transform!

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
end
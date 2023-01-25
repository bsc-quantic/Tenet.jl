@testset "Tensor" begin
    using Tenet: Tensor, labels

    @testset "Constructor" begin
        tensor = Tensor(zeros(2, 2, 2), (:i, :j, :k))
        @test labels(tensor) == (:i, :j, :k)
        @test labels(Tensor(zeros(2, 2, 2), [:i, :j, :k])) == (:i, :j, :k)
    end

    @testset "Broadcasting" begin
        data = rand(2, 2, 2)
        @test begin
            tensor = Tensor(data, (:a, :b, :c))
            tensor = tensor .+ one(eltype(tensor))

            parent(tensor) == data .+ one(eltype(tensor))
        end

        @test begin
            tensor = Tensor(data, (:a, :b, :c))
            tensor = sin.(tensor)

            parent(tensor) == sin.(data)
        end
    end
end
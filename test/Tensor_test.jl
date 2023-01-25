using Tenet: Tensor

@testset "Tensor" begin
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
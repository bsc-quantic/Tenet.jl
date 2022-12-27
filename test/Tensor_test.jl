using Tenet: Tensor

@testset "Tensor" verbose = true begin
    A = rand(2, 2, 2)
    @testset "Broadcasting" begin
        @test begin
            t = Tensor(A, (:a, :b, :c))
            t = t .+ one(eltype(t))

            parent(t) == A .+ one(eltype(t))
        end

        @test begin
            t = Tensor(A, (:a, :b, :c))
            t = sin.(t)

            parent(t) == sin.(A)
        end
    end
end
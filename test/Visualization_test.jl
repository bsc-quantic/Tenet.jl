@testset "Visualization" begin
    using CairoMakie
    using Tenet
    using NetworkLayout: Spring

    @testset "plot`" begin
        tn = TensorNetwork([
            Tensor(rand(2, 2, 2, 2), (:x, :y, :z, :t)),
            Tensor(rand(2, 2), (:x, :y)),
            Tensor(rand(2), (:x))
            ])

        @test plot(tn) isa Any
        @test plot(tn; labels=true) isa Any
        @test plot(tn; layout=Spring(dim=3)) isa Any

        @test begin
            f = Figure()
            plot!(f[1,1], tn) isa Any
        end

        @test begin
            f = Figure()
            plot!(f[1,1], tn) isa Any
        end

        @test begin
            f = Figure()
            plot!(f[1,1], tn) isa Any
        end
    end
end
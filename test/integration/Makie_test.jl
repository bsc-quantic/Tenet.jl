@testset "Visualization" begin
    using CairoMakie
    using NetworkLayout: Spring

    tensors = Tensor[Tensor(rand(2, 2, 2, 2), (:x, :y, :z, :t)), Tensor(rand(2, 2), (:x, :y)), Tensor(rand(2), (:x,))]
    tn = TensorNetwork(tensors)

    @testset "plot!" begin
        f = Figure()
        @testset "(default)" plot!(f[1, 1], tn)
        @testset "with labels" plot!(f[1, 1], tn; labels = true)
        @testset "3D" plot!(f[1, 1], tn; layout = Spring(dim = 3))
    end

    @testset "plot" begin
        @testset "(default)" plot(tn)
        @testset "with labels" plot(tn; labels = true)
        @testset "3D" plot(tn; layout = Spring(dim = 3))
    end
end

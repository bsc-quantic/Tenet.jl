@testset "Visualization" begin
    using GraphMakie
    using CairoMakie
    using NetworkLayout: Spring

    tensors = [Tensor(rand(2, 2, 2, 2), (:x, :y, :z, :t)), Tensor(rand(2, 2), (:x, :y)), Tensor(rand(2), (:x,))]
    tn = TensorNetwork(tensors)

    @testset "plot!" begin
        f = Figure()
        @testset "(default)" plot!(f[1, 1], tn)
        @testset "with labels" plot!(f[1, 1], tn; labels=true)
        @testset "with sizes" plot!(f[1, 1], tn; node_size=[5, 10, 15])
        @testset "with colors" plot!(f[1, 1], tn; node_color=[:red, :green, :blue])
        @testset "3D" plot!(f[1, 1], tn; layout=Spring(; dim=3))
    end

    @testset "plot" begin
        @testset "(default)" plot(tn)
        @testset "with labels" plot(tn; labels=true)
        @testset "3D" plot(tn; layout=Spring(; dim=3))
    end
end

@testset "Visualization" begin
    using GraphMakie
    using CairoMakie
    using NetworkLayout: Spring

    tensors = [Tensor(rand(2, 2, 2, 2), (:x, :y, :z, :t)), Tensor(rand(2, 2), (:x, :y)), Tensor(rand(2), (:x,))]
    tn = TensorNetwork(tensors)

    @testset "plot!" begin
        f = Figure()
        @testset "(default)" graphplot!(f[1, 1], tn)
        @testset "with labels" graphplot!(f[1, 1], tn; labels=true)
        @testset "with sizes" graphplot!(f[1, 1], tn; node_size=[5, 10, 15])
        @testset "with colors" graphplot!(f[1, 1], tn; node_color=[:red, :green, :blue])
        @testset "3D" graphplot!(f[1, 1], tn; layout=Spring(; dim=3))
    end

    @testset "plot" begin
        @testset "(default)" graphplot(tn)
        @testset "with labels" graphplot(tn; labels=true)
        @testset "3D" graphplot(tn; layout=Spring(; dim=3))
    end
end

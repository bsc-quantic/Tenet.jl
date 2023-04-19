@testset "Visualization" begin
    using CairoMakie
    using NetworkLayout: Spring

    tn = TensorNetwork([
        Tensor(rand(2, 2, 2, 2), (:x, :y, :z, :t)),
        Tensor(rand(2, 2), (:x, :y)),
        Tensor(rand(2), (:x,)),
    ])

    @testset "plot!" begin
        f = Figure()
        @testset "(default)" plot!(f[1, 1], path)
        @testset "with labels" plot!(f[1, 1], path; labels = true)
        @testset "3D" plot!(f[1, 1], path; layout = Spring(dim = 3))
    end

    @testset "plot" begin
        @testset "(default)" plot(path)
        @testset "with labels" plot(path; labels = true)
        @testset "3D" plot(path; layout = Spring(dim = 3))
    end
end
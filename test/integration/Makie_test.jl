@testset "Visualization" begin
    using CairoMakie
    using Tenet
    using NetworkLayout: Spring
    using Makie: FigureAxisPlot, AxisPlot

    @testset "plot`" begin
        tn = TensorNetwork([
            Tensor(rand(2, 2, 2, 2), (:x, :y, :z, :t)),
            Tensor(rand(2, 2), (:x, :y)),
            Tensor(rand(2), (:x,)),
        ])

        @test plot(tn) isa FigureAxisPlot
        @test plot(tn; labels = true) isa FigureAxisPlot
        @test plot(tn; layout = Spring(dim = 3)) isa FigureAxisPlot

        @test begin
            f = Figure()
            plot!(f[1, 1], tn) isa AxisPlot
        end

        @test begin
            f = Figure()
            plot!(f[1, 1], tn; labels = true) isa AxisPlot
        end

        @test begin
            f = Figure()
            plot!(f[1, 1], tn; layout = Spring(dim = 3)) isa AxisPlot
        end
    end
end
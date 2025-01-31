@safetestset "Visualization" begin
    using Test
    using Tenet
    using GraphMakie
    using CairoMakie
    using NetworkLayout: Spring

    tensors = [Tensor(rand(2, 2, 2, 2), (:x, :y, :z, :t)), Tensor(rand(2, 2), (:x, :y)), Tensor(rand(2), (:x,))]
    tn = TensorNetwork(tensors)

    @testset "plot!" begin
        f = Figure()
        @testset "(default)" begin
            @test try
                graphplot!(f[1, 1], tn)
                return true
            catch e
                @warn e
                return false
            end
        end

        @testset "with labels" begin
            @test try
                graphplot!(f[1, 1], tn; labels=true)
                return true
            catch e
                @warn e
                return false
            end
        end

        @testset "with sizes" begin
            @test try
                graphplot!(f[1, 1], tn; node_size=[5, 10, 15])
                return true
            catch e
                @warn e
                return false
            end
        end

        @testset "with colors" begin
            @test try
                graphplot!(f[1, 1], tn; node_color=[:red, :green, :blue])
                return true
            catch e
                @warn e
                return false
            end
        end

        @testset "3D" begin
            @test try
                graphplot!(f[1, 1], tn; layout=Spring(; dim=3))
                return true
            catch e
                @warn e
                return false
            end
        end
    end

    @testset "plot" begin
        @testset "(default)" begin
            try
                graphplot(tn)
                return true
            catch e
                @warn e
                return false
            end
        end

        @testset "with labels" begin
            try
                graphplot(tn; labels=true)
                return true
            catch e
                @warn e
                return false
            end
        end

        @testset "3D" begin
            try
                graphplot(tn; layout=Spring(; dim=3))
                return true
            catch e
                @warn e
                return false
            end
        end
    end
end

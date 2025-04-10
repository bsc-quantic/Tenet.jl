using Test
using Tenet

@testset "constructor" begin
    @testset let
        ψ = rand(MPS; n=5, maxdim=8)
        stn = stack(ψ)

        @test ntensors(stn) == 5
        @test ninds(stn) == 9
        @test issetequal(sites(stn), [site"1", site"2", site"3", site"4", site"5"])
        @test nlayers(stn) == 1
        @test layer(stn, 1) isa MPS
    end

    @testset let
        ψ = rand(MPS; n=5, maxdim=8)
        stn = stack(ψ')

        @test ntensors(stn) == 5
        @test ninds(stn) == 9
        @test issetequal(sites(stn), [site"1'", site"2'", site"3'", site"4'", site"5'"])
        @test nlayers(stn) == 1
        @test layer(stn, 1) isa MPS
    end

    @testset let
        ψ = rand(MPS; n=5, maxdim=8)
        stn = stack(ψ, ψ')

        @test ntensors(stn) == 10
        @test ninds(stn) == 13
        @test isempty(sites(stn))
        @test nlayers(stn) == 2
        @test layer(stn, 1) isa MPS
        @test layer(stn, 2) isa MPS

        layer1 = layer(stn, 1)
        layer2 = layer(stn, 2)

        for site in sites(layer1; set=:outputs)
            @test inds(layer1; at=site) == inds(layer2; at=site')
        end
    end

    @testset let
        ψ = rand(MPS; n=5, maxdim=8)
        stn = stack(ψ)
        push!(stn, ψ')

        @test ntensors(stn) == 10
        @test ninds(stn) == 13
        @test isempty(sites(stn))
        @test nlayers(stn) == 2
        @test layer(stn, 1) isa MPS
        @test layer(stn, 2) isa MPS

        layer1 = layer(stn, 1)
        layer2 = layer(stn, 2)

        for site in sites(layer1; set=:outputs)
            @test inds(layer1; at=site) == inds(layer2; at=site')
        end
    end

    @testset let
        ψ = rand(MPS; n=5, maxdim=8)
        H = Product([rand(2, 2) for _ in 1:5])
        stn = stack(ψ, H)

        @test ntensors(stn) == 10
        @test ninds(stn) == 14
        @test issetequal(sites(stn), [site"1", site"2", site"3", site"4", site"5"])
        @test nlayers(stn) == 2
        @test layer(stn, 1) isa MPS
        @test layer(stn, 2) isa ProductOperator

        layer1 = layer(stn, 1)
        layer2 = layer(stn, 2)

        for site in sites(layer1; set=:outputs)
            @test inds(layer1; at=site) == inds(layer2; at=site')
        end
    end

    @testset let
        ψ = rand(MPS; n=5, maxdim=8)
        H = Product([rand(2, 2) for _ in 1:5])
        stn = stack(H, ψ')

        @test ntensors(stn) == 10
        @test ninds(stn) == 14
        @test issetequal(sites(stn), [site"1'", site"2'", site"3'", site"4'", site"5'"])
        @test nlayers(stn) == 2
        @test layer(stn, 1) isa ProductOperator
        @test layer(stn, 2) isa MPS

        layer1 = layer(stn, 1)
        layer2 = layer(stn, 2)

        for site in sites(layer1; set=:outputs)
            @test inds(layer1; at=site) == inds(layer2; at=site')
        end
    end

    @testset let
        ψ = rand(MPS; n=5, maxdim=8)
        H = Product([rand(2, 2) for _ in 1:5])
        stn = stack(ψ, H, ψ')

        @test ntensors(stn) == 15
        @test ninds(stn) == 18
        @test isempty(sites(stn))
        @test nlayers(stn) == 3
        @test layer(stn, 1) isa MPS
        @test layer(stn, 2) isa ProductOperator
        @test layer(stn, 3) isa MPS

        layer1 = layer(stn, 1)
        layer2 = layer(stn, 2)
        layer3 = layer(stn, 3)

        for site in sites(layer1; set=:outputs)
            @test inds(layer1; at=site) == inds(layer2; at=site')
        end

        for site in sites(layer2; set=:outputs)
            @test inds(layer2; at=site) == inds(layer3; at=site')
        end
    end
end

@testset "adjoint!" begin
    @testset let
        ψ = rand(MPS; n=5, maxdim=8)
        stn = stack(ψ) |> adjoint

        @test issetequal(sites(stn), [site"1'", site"2'", site"3'", site"4'", site"5'"])
    end

    @testset let
        ψ = rand(MPS; n=5, maxdim=8)
        H = Product([rand(2, 2) for _ in 1:5])
        stn = stack(ψ, H)
        stn_dual = adjoint(stn)

        @test issetequal(sites(stn_dual), [site"1'", site"2'", site"3'", site"4'", site"5'"])

        # check that layers are reversed
        @test layer(stn_dual, 1) == adjoint(layer(stn, 2))
        @test layer(stn_dual, 2) == adjoint(layer(stn, 1))
    end

    @testset let
        ψ = rand(MPS; n=5, maxdim=8)
        H = Product([rand(2, 2) for _ in 1:5])
        stn = stack(H, ψ')
        stn_dual = adjoint(stn)

        @test issetequal(sites(stn_dual), [site"1", site"2", site"3", site"4", site"5"])

        # check that layers are reversed
        @test layer(stn_dual, 1) == adjoint(layer(stn, 2))
        @test layer(stn_dual, 2) == adjoint(layer(stn, 1))
    end

    @testset let
        ψ = rand(MPS; n=5, maxdim=8)
        H = Product([rand(2, 2) for _ in 1:5])
        stn = stack(ψ, H, ψ')
        stn_dual = adjoint(stn)

        @test isempty(sites(stn_dual))

        # check that layers are reversed
        @test layer(stn_dual, 1) == adjoint(layer(stn, 3))
        @test layer(stn_dual, 2) == adjoint(layer(stn, 2))
        @test layer(stn_dual, 3) == adjoint(layer(stn, 1))
    end
end

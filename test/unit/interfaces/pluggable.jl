using Test
using Tenet

@testset "align!" begin
    @testset let
        ψ = MPS([rand(2, 2), rand(2, 2, 2), rand(2, 2)])
        H = MPO([rand(2, 2, 2), rand(2, 2, 2, 2), rand(2, 2, 2)])

        Tenet.resetinds!(H)
        align!(ψ => H)

        for site in sites(ψ; set=:outputs)
            @test inds(ψ; at=site) == inds(H; at=site')
        end
    end

    @testset "state with more lanes than operator" begin
        # MPS-like tensor network with 4 sites
        mps4sites = rand(MPS; n=4, maxdim=2)

        # MPO-like tensor network with 3 sites
        mpo3sites = rand(MPO; n=3, maxdim=2)

        Tenet.resetinds!(mpo3sites)
        align!(mps4sites => mpo3sites)

        for lane in lanes(mpo3sites)
            @test inds(mps4sites; at=Site(lane)) == inds(mpo3sites; at=Site(lane; dual=true))
        end
    end

    @testset "state with less lanes than operator" begin
        # MPS-like tensor network with 3 sites
        mps3sites = rand(MPS; n=3, maxdim=2)

        # MPO-like tensor network with 4 sites
        mpo4sites = rand(MPO; n=4, maxdim=2)

        Tenet.resetinds!(mpo4sites)
        align!(mps3sites => mpo4sites)

        for site in sites(mps3sites; set=:outputs)
            @test inds(mps3sites; at=site) == inds(mpo4sites; at=site')
        end
    end
end

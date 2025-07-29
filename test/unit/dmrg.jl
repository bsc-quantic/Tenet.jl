using Test
using Tenet
using Tenet.DMRG: dmrg

n = 10
H = Tenet.Models.ising_1d_mpo(n, 1.0, 1.0)

@testset "1-site DMRG" begin
    # look for the closest product state to the ground state
    @testset let
        ψ = convert(MPS, ProductState(fill([1, 0], n)))
        ϕ, energy = dmrg(ψ, H, 10; method=Tenet.DMRG.Dmrg1())
        @test energy / n ≈ -1.1902477482849715 atol = 1e-4
    end

    @testset let
        delta = zeros(ComplexF64, 2, 2, 2)
        delta[1, 1, 1] = 1.0
        delta[2, 2, 2] = 1.0
        ψ = MPS([[1 0; 0 1], fill(delta, n - 2)..., [1 0; 0 1]])

        ϕ, energy = dmrg(ψ, H, 10; method=Tenet.DMRG.Dmrg1(), verbosity=0)
        @test energy / n ≈ -1.2372129315563634 atol = 1e-4
    end
end

@testset "2-site DMRG" begin
    # look for the closest product state to the ground state
    @testset let
        ψ = convert(MPS, ProductState(fill([1, 0], n)))
        ϕ, energy = dmrg(ψ, H, 10; method=Tenet.DMRG.Dmrg2(), maxdim=1, verbosity=0)
        @test energy / n ≈ -1.1944098151375718 atol = 1e-4
    end

    @testset let
        ψ = convert(MPS, ProductState(fill([1, 0], n)))
        ϕ, energy = dmrg(ψ, H, 10; method=Tenet.DMRG.Dmrg2(), maxdim=32, verbosity=0)
        @test energy / n ≈ -1.2381489999654788 atol = 1e-4
    end
end

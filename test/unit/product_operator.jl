using Test
using Tenet
using LinearAlgebra

tn = ProductOperator([Float64[1 2; 3 4], Float64[5 6; 7 8], Float64[9 10; 11 12]])

@test issetequal(sites(tn), [site"1", site"2", site"3"])
@test isempty(bonds(tn))
@test issetequal(plugs(tn), [plug"1", plug"2", plug"3", plug"1'", plug"2'", plug"3'"])

@test parent(tn[site"1"]) ≈ Float64[1 2; 3 4]
@test parent(tn[site"2"]) ≈ Float64[5 6; 7 8]
@test parent(tn[site"3"]) ≈ Float64[9 10; 11 12]

@testset "copy" begin
    tn_copy = copy(tn)
    @test tn_copy !== tn
    @test all(∈(tn), all_tensors(tn_copy))
end

@testset "zero" begin
    tn_zero = zero(tn)
    @test all(iszero, arrays(tn_zero))
end

@testset "norm" begin
    @test norm(tn) ≈ prod(norm, arrays(tn))
end

@testset "normalize" begin
    @testset let tn = normalize(tn)
        @test norm(tn) ≈ 1
    end
end

@testset "adjoint" begin
    @testset let tn = copy(tn)
        tn[plug"1"] = Index(:i)
        tn[plug"1'"] = Index(:j)

        tn_adj = adjoint(tn)
        @test issetequal(plugs(tn_adj), [plug"1", plug"2", plug"3", plug"1'", plug"2'", plug"3'"])

        @test tn_adj[plug"1"] == Index(:j)
        @test tn_adj[plug"1'"] == Index(:i)
    end
end

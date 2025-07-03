using Test
using Tenet
using LinearAlgebra

tn = ProductState([ones(2), 2ones(2), 3ones(2)])

@test issetequal(sites(tn), [site"1", site"2", site"3"])
@test isempty(bonds(tn))
@test issetequal(plugs(tn), [plug"1", plug"2", plug"3"])

@test parent(tn[site"1"]) == ones(2)
@test parent(tn[site"2"]) == 2ones(2)
@test parent(tn[site"3"]) == 3ones(2)

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
    tn_adj = adjoint(tn)
    @test issetequal(plugs(tn_adj), [plug"1'", plug"2'", plug"3'"])
end

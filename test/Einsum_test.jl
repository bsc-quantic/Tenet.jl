using Tenet: parse, OuterProduct, InnerProduct, Trace, Permutation
import Permutations: Permutation as Permutator

@testset "Einsum" verbose = true begin
    @testset "Parsing" verbose = true begin
        @testset "InnerProduct" begin
            @test parse(InnerProduct, [:m, :n], [:m, :k], [:k, :n]) == [:k]
            @test parse(InnerProduct, [:n, :m], [:m, :k], [:k, :n]) == [:k]
            @test parse(InnerProduct, [:m, :k, :n], [:m, :k], [:k, :n]) == []
            @test parse(InnerProduct, [:m, :n], [:m], [:n]) == []
        end

        @testset "OuterProduct" begin
            @test parse(OuterProduct, [:m, :n], [:m, :k], [:k, :n]) == [:m, :n]
            @test parse(OuterProduct, [:n, :m], [:m, :k], [:k, :n]) == [:m, :n]
            @test parse(OuterProduct, [:m, :k, :n], [:m, :k], [:k, :n]) == [:m, :n]
            @test parse(OuterProduct, [:m, :n], [:m], [:n]) == [:m, :n]
        end

        @testset "Trace" begin
            @test parse(Trace, [:i], [:i, :i]) == [:i]
            @test parse(Trace, [:m, :n], [:m, :n]) == []
            @test parse(Trace, [:m, :k, :n], [:m, :k, :n, :k]) == [:k]
        end

        @testset "Permutation" begin
            @test parse(Permutation, [:a, :b], [:a, :b]) == Permutator([1, 2])
            @test parse(Permutation, [:b, :a], [:a, :b]) == Permutator([2, 1])
        end
    end
end
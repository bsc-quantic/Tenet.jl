@testset "Helpers" begin
    @testset "Sequence" begin
        @test Vector <: Tenet.Sequence
        @test (Vector{T} where {T}) <: Tenet.Sequence{T} where {T}
        @test NTuple <: Tenet.Sequence
        @test (NTuple{N} where {N}) <: Tenet.Sequence
        @test (NTuple{N,T} where {N,T}) <: Tenet.Sequence where {T}
    end

    @testset "RingPeek" begin
        using Tenet: ringpeek

        it = ringpeek([0 1 2])
        @test length(it) == 3
        @test collect(it) == [(0, 1), (1, 2), (2, 0)]
    end
end
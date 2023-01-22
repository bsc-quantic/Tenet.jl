@testset "Helpers" begin
    @test Vector <: Tenet.Sequence
    @test (Vector{T} where {T}) <: Tenet.Sequence{T} where {T}
    @test NTuple <: Tenet.Sequence
    @test (NTuple{N} where {N}) <: Tenet.Sequence
    @test (NTuple{N,T} where {N,T}) <: Tenet.Sequence where {T}
end
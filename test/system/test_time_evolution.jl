using Test
using Tenet

@testset "case: mps time-evolution through circuit" begin
    ψ = MPS("00")

    hadamard = 1 / sqrt(2) * [1 1; 1 -1]
    cnot = [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]

    circuit = Circuit()
    push!(circuit, Gate(hadamard, [site"1", site"1'"]))
    push!(circuit, Gate(hadamard, [site"2", site"2'"]))
    push!(circuit, Gate(cnot, [site"1", site"2", site"1'", site"2'"]))

    evolve!(ψ, circuit)

    ϕ = dense(1//2 * [1 0; 0 1], [site"1", site"2"])
    @test overlap(ψ, ϕ) ≈ 1.0
end

@testset "case: mps time-evolution through mpo" begin
    ψ = MPS("00")

    mpo = evolve!(ψ, mpo)

    ϕ = dense()
    @test overlap(ψ, ϕ) ≈ 1.0
end

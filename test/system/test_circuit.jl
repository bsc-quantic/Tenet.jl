using Test
using Tenet

@testset "case: collapse circuit into mpo" begin
    hadamard = 1 / sqrt(2) * [1 1; 1 -1]
    cnot = [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]

    circuit = Circuit()
    push!(circuit, Gate(hadamard, [site"1", site"1'"]))
    push!(circuit, Gate(hadamard, [site"2", site"2'"]))
    push!(circuit, Gate(cnot, [site"1", site"2", site"1'", site"2'"]))

    mpo = convert(MPO, circuit)

    @test mpo isa MPO
end

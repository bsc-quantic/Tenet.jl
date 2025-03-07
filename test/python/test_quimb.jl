@testset "quimb" begin
    qtn = pyimport("quimb.tensor")

    # NOTE quimb.circuit.Circuit splits gates by default
    qc = qtn.Circuit(3; gate_opts=Dict(["contract" => false]))
    gates = [("H", 0), ("H", 1), ("CNOT", 1, 2), ("CNOT", 0, 2), ("H", 0), ("H", 1), ("H", 2)]
    qc.apply_gates(gates)

    tn = convert(Quantum, qc)
    @test issetequal(sites(tn; set=:inputs), adjoint.(Site.([0, 1, 2])))
    @test issetequal(sites(tn; set=:outputs), Site.([0, 1, 2]))
    @test Tenet.ntensors(tn) == 7

    tn = convert(Quantum, qc.psi)
    @test isempty(sites(tn; set=:inputs))
    @test issetequal(sites(tn; set=:outputs), Site.([0, 1, 2]))
    @test Tenet.ntensors(tn) == 10
end

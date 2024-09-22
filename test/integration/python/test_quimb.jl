@testset "quimb" begin
    using PythonCall
    qtn = pyimport("quimb.tensor")

    qc = qtn.Circuit(3)
    gates = [("H", 0), ("H", 1), ("CNOT", 1, 2), ("CNOT", 0, 2), ("H", 0), ("H", 1), ("H", 2)]
    qc.apply_gates(gates)

    tn = Tenet.Quantum(qc)
    @test issetequal(sites(tn; set=:inputs), adjoint.(Site.([0, 1, 2])))
    @test issetequal(sites(tn; set=:outputs), Site.([0, 1, 2]))
    @test Tenet.ntensors(tn) == 7

    tn = Tenet.Quantum(qc.psi)
    @test isempty(sites(tn; set=:inputs))
    @test issetequal(sites(tn; set=:outputs), Site.([0, 1, 2]))
    @test Tenet.ntensors(tn) == 10
end

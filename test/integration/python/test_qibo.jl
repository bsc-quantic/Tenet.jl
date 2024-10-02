@testset "qibo" begin
    using PythonCall
    qiskit = pyimport("qibo")

    circuit = qibo.Circuit(3)
    circuit.add(qibo.gates.H(0))
    circuit.add(qibo.gates.H(1))
    circuit.add(qibo.gates.CNOT(1,2))
    circuit.add(qibo.gates.CNOT(0,2))
    circuit.add(qibo.gates.H(0))
    circuit.add(qibo.gates.H(1))
    circuit.add(qibo.gates.H(2))

    tn = Tenet.Quantum(circuit)
    @test issetequal(sites(tn; set=:inputs), adjoint.(Site.([1, 2, 3])))
    @test issetequal(sites(tn; set=:outputs), Site.([1, 2, 3]))
    @test Tenet.ntensors(tn) == 7
end
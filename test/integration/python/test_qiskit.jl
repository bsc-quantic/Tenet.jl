@testset "qiskit" begin
    using PythonCall
    qiskit = pyimport("qiskit")

    circuit = qiskit.QuantumCircuit(3)
    circuit.h(0)
    circuit.h(1)
    circuit.cx(1, 2)
    circuit.cx(0, 2)
    circuit.h(0)
    circuit.h(1)
    circuit.h(2)

    tn = convert(Quantum, circuit)
    @test issetequal(sites(tn; set=:inputs), adjoint.(Site.([1, 2, 3])))
    @test issetequal(sites(tn; set=:outputs), Site.([1, 2, 3]))
    @test Tenet.ntensors(tn) == 7
end

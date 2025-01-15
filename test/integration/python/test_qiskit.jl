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

    tn = convert(Circuit, circuit)
    @test tn isa Circuit
    @test issetequal(sites(circ; set=:inputs), Site.([1, 2, 3]; dual=true))
    @test issetequal(sites(circ; set=:outputs), Site.([1, 2, 3]))
    @test Tenet.ntensors(circ) == 7
    @test issetequal(
        moments(circ), [Moment.(Ref(Lane(1)), 1:4)..., Moment.(Ref(Lane(2)), 1:4)..., Moment.(Ref(Lane(3)), 1:4)...]
    )
end

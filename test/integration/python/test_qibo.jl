@testset "qibo" begin
    using PythonCall
    qibo = pyimport("qibo")

    circuit = qibo.Circuit(3)
    circuit.add(qibo.gates.H(0))
    circuit.add(qibo.gates.H(1))
    circuit.add(qibo.gates.CNOT(1, 2))
    circuit.add(qibo.gates.CNOT(0, 2))
    circuit.add(qibo.gates.H(0))
    circuit.add(qibo.gates.H(1))
    circuit.add(qibo.gates.H(2))

    circ = convert(Circuit, Val(:qibo), circuit)
    @test circ isa Circuit
    @test issetequal(sites(circ; set=:inputs), Site.([0, 1, 2]; dual=true))
    @test issetequal(sites(circ; set=:outputs), Site.([0, 1, 2]))
    @test Tenet.ntensors(circ) == 7
    @test issetequal(
        moments(circ), [Tenet.Moment.(Ref(Lane(0)), 1:4)..., Tenet.Moment.(Ref(Lane(1)), 1:4)..., Tenet.Moment.(Ref(Lane(2)), 1:4)...]
    )
end

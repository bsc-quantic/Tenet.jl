@testset "cirq" begin
    using PythonCall
    cirq = pyimport("cirq")

    @testset "LineQubit" begin
        circuit = cirq.Circuit()
        qubits = cirq.LineQubit.range(3)
        circuit.append(cirq.H(qubits[0]))
        circuit.append(cirq.H(qubits[1]))
        circuit.append(cirq.CNOT(qubits[1], qubits[2]))
        circuit.append(cirq.CNOT(qubits[0], qubits[2]))
        circuit.append(cirq.H(qubits[0]))
        circuit.append(cirq.H(qubits[1]))
        circuit.append(cirq.H(qubits[2]))

        circ = convert(Circuit, circuit)
        @test issetequal(sites(circ; set=:inputs), Site.([0, 1, 2]; dual=true))
        @test issetequal(sites(circ; set=:outputs), Site.([0, 1, 2]))
        @test Tenet.ntensors(circ) == 7
        @test issetequal(
            moments(circ), [Moment.(Ref(Lane(0)), 1:4)..., Moment.(Ref(Lane(1)), 1:4)..., Moment.(Ref(Lane(2)), 1:4)...]
        )
    end

    @testset "GridQubit" begin
        circuit = cirq.Circuit()
        qubits = cirq.GridQubit.rect(3, 1)
        circuit.append(cirq.H(qubits[0]))
        circuit.append(cirq.H(qubits[1]))
        circuit.append(cirq.CNOT(qubits[1], qubits[2]))
        circuit.append(cirq.CNOT(qubits[0], qubits[2]))
        circuit.append(cirq.H(qubits[0]))
        circuit.append(cirq.H(qubits[1]))
        circuit.append(cirq.H(qubits[2]))

        circ = convert(Circuit, circuit)
        @test issetequal(sites(circ; set=:inputs), Site.([(0, 0), (1, 0), (2, 0)]; dual=true))
        @test issetequal(sites(circ; set=:outputs), Site.([(0, 0), (1, 0), (2, 0)]))
        @test Tenet.ntensors(circ) == 7
        @test issetequal(
            moments(circ),
            [Moment.(Ref(Lane(0, 0)), 1:4)..., Moment.(Ref(Lane(1, 0)), 1:4)..., Moment.(Ref(Lane(2, 0)), 1:4)...],
        )
    end
end

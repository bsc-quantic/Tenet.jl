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

        tn = convert(Quantum, circuit)
        @test issetequal(sites(tn; set=:inputs), adjoint.(Site.([0, 1, 2])))
        @test issetequal(sites(tn; set=:outputs), Site.([0, 1, 2]))
        @test Tenet.ntensors(tn) == 7
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

        tn = convert(Quantum, circuit)
        @test issetequal(sites(tn; set=:inputs), adjoint.(Site.([(0, 0), (1, 0), (2, 0)])))
        @test issetequal(sites(tn; set=:outputs), Site.([(0, 0), (1, 0), (2, 0)]))
        @test Tenet.ntensors(tn) == 7
    end
end

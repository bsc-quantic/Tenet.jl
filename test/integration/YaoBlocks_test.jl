@testset "YaoBlocks" begin
    using YaoBlocks

    # NOTE qubit #3 left empty on purpose
    circuit = chain(3, put(1 => X), cnot(1, 2))
    tn = Quantum(circuit)

    @test issetequal(sites(tn), [site"1", site"2", site"1'", site"2'"])
    @test Tenet.ntensors(tn) == 2

    @testset "GHZ Circuit" begin
        circuit_GHZ = chain(
            n_qubits,
            put(1=>Yao.H),
            Yao.control(1, 2=>Yao.X),
            Yao.control(2, 3=>Yao.X)
        )

        quantum_circuit = Quantum(circuit_GHZ)

        zeros = Quantum(Product(fill([1, 0], n_qubits))) #|000>
        ones = Quantum(Product(fill([0, 1], n_qubits))) #|111>

        expected_value = Tenet.contract(merge(zeros, quantum_circuit, ones')) # <111|circuit|000>
        @test only(expected_value) ≈ 1/√2

        SV_Yao = apply!(zero_state(n_qubits), circuit_GHZ) # circuit|000>
        @test only(statevec(ArrayReg(bit"111"))' * statevec(SV_Yao)) ≈ 1/√2
    end
end

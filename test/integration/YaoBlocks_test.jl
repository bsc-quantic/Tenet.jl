using YaoBlocks

@testset "YaoBlocks" begin
    # NOTE qubit #3 left empty on purpose
    @testset let yaocirc = chain(3, put(1 => X), cnot(1, 2))
        circuit = convert(Circuit, yaocirc)

        @test issetequal(sites(circuit), [site"1", site"2", site"1'", site"2'"])
        @test Tenet.ntensors(circuit) == 2
    end

    @testset "GHZ Circuit" begin
        n_qubits = 3
        yaocirc = chain(n_qubits, put(1 => Yao.H), Yao.control(1, 2 => Yao.X), Yao.control(2, 3 => Yao.X))
        circuit = convert(Circuit, yaocirc)

        zeros = Quantum(Product(fill([1, 0], n_qubits))) #|000>
        ones = Quantum(Product(fill([0, 1], n_qubits))) #|111>

        expected_value = Tenet.contract(merge(zeros, Quantum(circuit), ones')) # <111|circuit|000>
        @test only(expected_value) ≈ 1 / √2

        yaosv = apply!(zero_state(n_qubits), yaocirc) # circuit|000>
        @test only(statevec(ArrayReg(YaoBlocks.bit"111"))' * statevec(yaosv)) ≈ 1 / √2
    end

    @testset "two-qubit gate" begin
        U = matblock(rand(ComplexF64, 4, 4); tag="U")
        yaocirc = chain(2, put((1, 2) => U))
        psi = zero_state(2)
        apply!(psi, yaocirc)

        circuit = convert(Circuit, yaocirc)
        zeros = Quantum(Product(fill([1, 0], 2))) #|00>
        ones = Quantum(Product(fill([0, 1], 2))) #|11>

        expected_value = Tenet.contract(merge(zeros, Quantum(circuit), ones')) # <11|circuit|00>

        yaosv = apply!(zero_state(2), yaocirc) # circuit|00>

        @test only(expected_value) ≈ only(statevec(ArrayReg(YaoBlocks.bit"11"))' * statevec(yaosv))
    end
end

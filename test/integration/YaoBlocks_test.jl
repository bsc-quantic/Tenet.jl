using Yao
using YaoBlocks

@testset "YaoBlocks" begin
    # NOTE qubit #3 left empty on purpose
    @testset let yaocirc = chain(3, put(1 => X), cnot(1, 2))
        circuit = convert(Circuit, yaocirc)

        @test issetequal(sites(circuit), [site"1", site"2", site"1'", site"2'"])
        @test Tenet.ntensors(circuit) == 2
    end

    @testset "GHZ" begin
        n_qubits = 3
        yaocirc = chain(n_qubits, put(1 => H), Yao.control(1, 2 => X), Yao.control(2, 3 => X))
        circuit = convert(Circuit, yaocirc)

        zerost = Quantum(Product(fill([1, 0], n_qubits))) #|000>
        onest = Quantum(Product(fill([0, 1], n_qubits))) #|111>

        expected_value = Tenet.contract(merge(zerost, Quantum(circuit), onest')) # Tenet <111|circuit|000>

        yaosv = apply!(zero_state(n_qubits), yaocirc) # circuit|000>
        @test only(expected_value) ≈ only(statevec(ArrayReg(bit"111"))' * statevec(yaosv)) ≈ 1 / √2 # Yao <111|circuit|000>
    end

    @testset "two-qubit dense gate" begin
        n_qubits = 2
        U = matblock(rand(ComplexF64, 4, 4); tag="U")
        yaocirc = chain(n_qubits, put((1, 2) => U))
        circuit = convert(Circuit, yaocirc)

        zerost = Quantum(Product(fill([1, 0], n_qubits))) #|00>
        onest = Quantum(Product(fill([0, 1], n_qubits))) #|11>

        expected_value = Tenet.contract(merge(zerost, Quantum(circuit), onest')) # Tenet <11|circuit|00>

        yaosv = apply!(zero_state(n_qubits), yaocirc) # circuit|00>
        @test only(expected_value) ≈ only(statevec(ArrayReg(bit"11"))' * statevec(yaosv)) # Yao <11|circuit|00>
    end
end

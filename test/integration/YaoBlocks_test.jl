@safetestset "YaoBlocks" begin
    using Test
    using Tenet
    using Yao
    using YaoBlocks

    # NOTE qubit #3 left empty on purpose
    @testset let yaocirc = chain(3, put(1 => X), cnot(1, 2))
        circuit = convert(Circuit, yaocirc)

        @test issetequal(sites(circuit), [site"1", site"2", site"1'", site"2'"])
        @test Tenet.ntensors(circuit) == 2
    end

    @testset "GHZ" begin
        n = 3
        yaocirc = chain(n, put(1 => Yao.H), Yao.control(1, 2 => Yao.X), Yao.control(2, 3 => Yao.X))
        circuit = convert(Circuit, yaocirc)

        # <111|circuit|000>
        zeros = Quantum(Product(fill([1, 0], n))) #|000>
        ones = Quantum(Product(fill([0, 1], n))) #|111>
        ampl111 = only(Tenet.contract(merge(zeros, Quantum(circuit), ones')))

        yaoampl111 = apply!(zero_state(n), yaocirc)[bit"111"]

        @test yaoampl111 ≈ ampl111 ≈ 1 / √2
    end

    @testset "two-qubit dense gate" begin
        n = 2
        U = matblock(rand(ComplexF64, 4, 4); tag="U")
        yaocirc = chain(2, put((1, 2) => U))

        # <11|circuit|00>
        circuit = convert(Circuit, yaocirc)
        zeros = Quantum(Product(fill([1, 0], 2))) #|00>
        ones = Quantum(Product(fill([0, 1], 2))) #|11>
        ampl11 = Tenet.contract(merge(zeros, Quantum(circuit), ones'))

        yaoampl11 = apply!(zero_state(n), yaocirc)[bit"11"]

        @test only(ampl11) ≈ yaoampl11
    end
end

@testset "YaoBlocks" begin
    using YaoBlocks

    # NOTE qubit #3 left empty on purpose
    circuit = chain(3, put(1 => X), cnot(1, 2))
    tn = Quantum(circuit)

    @test issetequal(sites(tn), [site"1", site"2", site"1'", site"2'"])
    @test Tenet.ntensors(tn) == 2
end

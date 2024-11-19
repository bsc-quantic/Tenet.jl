@testset "Quac" begin
    using Quac

    @testset "Gate" begin
        id_gate = 2
        gate = Quac.Z(id_gate)

        qgate = Quantum(gate)

        @test nsites(qgate; set=:inputs) == 1
        @test nsites(qgate; set=:outputs) == 1
        @test issetequal(sites(qgate), [Site(id_gate), Site(id_gate; dual=true)])
        @test socket(qgate) == Operator()
    end

    @testset "QFT" begin
        n = 3
        qftcirc = Quac.Algorithms.QFT(n)
        qftqtn = Quantum(qftcirc)

        # correct number of inputs and outputs
        @test nsites(qftqtn; set=:inputs) == n
        @test nsites(qftqtn; set=:outputs) == n
        @test socket(qftqtn) == Operator()

        # all open indices are sites
        siteinds = getindex.((qftqtn,), sites(qftqtn))
        @test issetequal(inds(TensorNetwork(qftqtn); set=:open), siteinds)

        # all inner indices are not sites
        # TODO too strict condition. remove?
        notsiteinds = setdiff(inds(TensorNetwork(qftqtn)), siteinds)
        @test_skip issetequal(inds(TensorNetwork(qftqtn); set=:inner), notsiteinds)
    end
end

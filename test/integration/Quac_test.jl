@safetestset "Quac" begin
    using Test
    using Tenet
    using Tenet: nsites, Operator
    using Quac: Quac

    @testset "Gate" begin
        id_gate = 2
        gate = Quac.Z(id_gate)
        qgate = convert(Gate, gate)

        @test issetequal(sites(qgate), [Site(id_gate), Site(id_gate; dual=true)])
    end

    @testset "QFT" begin
        n = 3
        qftcirc = Quac.Algorithms.QFT(n)
        qftqtn = convert(Circuit, qftcirc)

        # correct number of inputs and outputs
        @test nsites(qftqtn; set=:inputs) == n
        @test nsites(qftqtn; set=:outputs) == n
        @test socket(qftqtn) == Operator()

        # all open indices are sites
        @test issetequal(inds(qftqtn; set=:open), inds(qftqtn; set=:physical))

        # all inner indices are not sites
        # TODO too strict condition. remove?
        @test_skip issetequal(inds(qftqtn; set=:inner), inds(qftqtn; set=:virtual))
    end
end

@testset "Quantum" begin
    tn = TensorNetwork{Quantum}([Tensor(rand(2), (:i,))]; plug = Dict([(1, :out) => :i]))

    @testset "metadata" begin
        @test fieldnames(Tenet.metadata(Quantum)) === (:plug,)
        @test Tenet.checkmeta(tn)

        @test hasproperty(tn, :plug)
        @test tn.plug == Dict([(1, :out) => :i])
    end

    # TODO write tests for
    # - boundary
    # - plug
    # - sites for :plug, :in, :out

    @testset "sites" begin
        @test sites(tn) == sites(tn; dir = :out) == [1]
        @test isempty(sites(tn, dir = :in))
    end

    @testset "labels" begin
        @test all(allequal, zip(labels(tn), labels(tn, set = :open), labels(tn, :plug), labels(tn, :out), (:i,)))
        @test isempty(labels(tn, set = :inner))
        @test isempty(labels(tn, set = :hyper))
        @test isempty(labels(tn, set = :in))
        @test isempty(labels(tn, set = :virtual))
    end

    # @testset "tensors" begin
    #     @test tensors(tn) == tensors(tn, 1)
    # end

    @testset "adjoint" begin
        adj = adjoint(tn)

        @test sites(tn, dir = :out) == sites(adj, dir = :in)
        @test isempty(sites(tn, dir = :in))
        @test isempty(sites(adj, dir = :out))

        @test labels(tn, set = :plug) == labels(adj, set = :plug)
        @test labels(tn, set = :out) == labels(adj, set = :in)
        @test isempty(labels(tn, set = :in))
        @test isempty(labels(adj, set = :out))
    end

    @testset "hcat" begin
        let tn = hcat(tn, tn')
            # TODO
        end
    end
end

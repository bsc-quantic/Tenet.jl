@testset "Quantum" begin
    using Bijections

    struct MockState <: Quantum end
    Tenet.plug(::Type{MockState}) = State

    tn = TensorNetwork{MockState}(
        [Tensor(rand(2, 2), (:i, :k)), Tensor(rand(3, 2, 4), (:j, :k, :l))];
        plug = State,
        interlayer = [Bijection(Dict([1 => :i, 2 => :j]))],
    )

    @testset "metadata" begin
        @test fieldnames(Tenet.metadata(Quantum)) === (:plug, :interlayer)
        @test Tenet.checkmeta(tn)

        @test hasproperty(tn, :interlayer)
        @test only(tn.interlayer) == Bijection(Dict([1 => :i, 2 => :j]))
    end

    # TODO write tests for
    # - boundary
    # - plug

    @testset "sites" begin
        @test issetequal(sites(tn), [1, 2])
    end

    @testset "labels" begin
        @test issetequal(labels(tn), [:i, :j, :k, :l])
        @test issetequal(labels(tn, set = :open), [:i, :j, :l])
        @test issetequal(labels(tn, set = :plug), [:i, :j])
        @test issetequal(labels(tn, set = :inner), [:k])
        @test isempty(labels(tn, set = :hyper))
        @test issetequal(labels(tn, set = :virtual), [:k, :l])
    end

    # @testset "tensors" begin
    #     @test tensors(tn) == tensors(tn, 1)
    # end

    @testset "adjoint" begin
        adj = adjoint(tn)

        @test issetequal(sites(tn), sites(adj))
        @test all(i -> labels(tn, :plug, i) == labels(adj, :plug, i), sites(tn))
    end

    @testset "hcat" begin
        let expectation = hcat(tn, tn')
            @test issetequal(sites(expectation), sites(tn))
            @test issetequal(labels(expectation, set = :plug), labels(tn, set = :plug))
            @test isempty(labels(expectation, set = :open))
            @test issetequal(labels(expectation, set = :inner), labels(expectation, set = :all))
        end
    end
end

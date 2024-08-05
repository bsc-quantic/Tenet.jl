@testset "Quantum" begin
    using Tenet

    _tensors = Tensor[Tensor(zeros(2), [:i])]
    tn = TensorNetwork(_tensors)
    qtn = Quantum(tn, Dict(site"1" => :i))
    @test ninputs(qtn) == 0
    @test noutputs(qtn) == 1
    @test issetequal(sites(qtn), [site"1"])
    @test socket(qtn) == State(; dual=false)

    # forwarded methods to `TensorNetwork`
    @test TensorNetwork(qtn) == tn
    @test tensors(qtn) == _tensors

    _tensors = Tensor[Tensor(zeros(2), [:i])]
    tn = TensorNetwork(_tensors)
    qtn = Quantum(tn, Dict(site"1'" => :i))
    @test ninputs(qtn) == 1
    @test noutputs(qtn) == 0
    @test issetequal(sites(qtn), [site"1'"])
    @test socket(qtn) == State(; dual=true)

    _tensors = Tensor[Tensor(zeros(2, 2), [:i, :j])]
    tn = TensorNetwork(_tensors)
    qtn = Quantum(tn, Dict(site"1" => :i, site"1'" => :j))
    @test ninputs(qtn) == 1
    @test noutputs(qtn) == 1
    @test issetequal(sites(qtn), [site"1", site"1'"])
    @test socket(qtn) == Operator()

    _tensors = Tensor[Tensor(fill(0))]
    tn = TensorNetwork(_tensors)
    qtn = Quantum(tn, Dict())
    @test ninputs(qtn) == 0
    @test noutputs(qtn) == 0
    @test isempty(sites(qtn))
    @test socket(qtn) == Scalar()

    # detect errors
    _tensors = Tensor[Tensor(zeros(2), [:i]), Tensor(zeros(2), [:i])]
    tn = TensorNetwork(_tensors)
    @test_throws ErrorException Quantum(tn, Dict(site"1" => :j))
    @test_throws ErrorException Quantum(tn, Dict(site"1" => :i))

    @testset "reindex!" begin
        @testset "manual indices" begin
            _tensors = Tensor[Tensor(rand(2, 2), [:i, :j]), Tensor(rand(2, 2, 2), [:j, :k, :l]), Tensor(rand(2, 2), [:l, :m])]
            tn = TensorNetwork(_tensors)
            qtn = Quantum(tn, Dict(site"1" => :i, site"2" => :k, site"3" => :m)) # mps-like tensor network

            _tensors2 = Tensor[Tensor(rand(2, 2, 2), [:i, :j, :k]), Tensor(rand(2, 2, 2, 2), [:l, :m, :k, :n]), Tensor(rand(2, 2, 2), [:o, :p, :n])]
            tn2 = TensorNetwork(_tensors2)
            qtn2 = Quantum(tn2, Dict(site"1" => :i, site"1'" => :j, site"2" => :l, site"2'" => :m, site"3" => :o, site"3'" => :p)) # mpo-like tensor network

            T₁ = tensors(qtn; at=site"1")
            U₁ = tensors(qtn2; at=site"1")

            Tenet.@reindex! outputs(qtn) => inputs(qtn2)

            @test issetequal([qtn2.sites[i] for i in inputs(qtn2)], [qtn.sites[i] for i in outputs(qtn)])
        end

        @testset "regular indices" begin
            _tensors = Tensor[Tensor(rand(2, 2), [:A, :B]), Tensor(rand(2, 2, 2), [:C, :B, :D]), Tensor(rand(2, 2), [:E, :D])]
            tn = TensorNetwork(_tensors)
            qtn = Quantum(tn, Dict(site"1" => :A, site"2" => :C, site"3" => :E)) # mps-like tensor network

            _tensors2 = Tensor[Tensor(rand(2, 2, 2), [:A, :B, :C]), Tensor(rand(2, 2, 2, 2), [:D, :E, :C, :F]), Tensor(rand(2, 2, 2), [:F, :G, :H])]
            tn2 = TensorNetwork(_tensors2)
            qtn2 = Quantum(tn2, Dict(site"1" => :A, site"1'" => :B, site"2" => :D, site"2'" => :E, site"3" => :G, site"3'" => :H)) # mpo-like tensor network

            T₁ = tensors(qtn; at=site"1")
            U₁ = tensors(qtn2; at=site"1")

            Tenet.@reindex! outputs(qtn) => inputs(qtn2)

            @test issetequal([qtn2.sites[i] for i in inputs(qtn2)], [qtn.sites[i] for i in outputs(qtn)])
        end
    end
end

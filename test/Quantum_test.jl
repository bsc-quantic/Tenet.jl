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
end

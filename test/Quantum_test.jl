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
            # mps-like tensor network
            mps = Quantum(
                TensorNetwork(
                    Tensor[
                        Tensor(rand(2, 2), [:i, :j]), Tensor(rand(2, 2, 2), [:j, :k, :l]), Tensor(rand(2, 2), [:l, :m])
                    ],
                ),
                Dict(site"1" => :i, site"2" => :k, site"3" => :m),
            )

            # mpo-like tensor network
            mpo = Quantum(
                TensorNetwork(
                    Tensor[
                        Tensor(rand(2, 2, 2), [:i, :j, :k]),
                        Tensor(rand(2, 2, 2, 2), [:l, :m, :k, :n]),
                        Tensor(rand(2, 2, 2), [:o, :p, :n]),
                    ],
                ),
                Dict(site"1" => :i, site"1'" => :j, site"2" => :l, site"2'" => :m, site"3" => :o, site"3'" => :p),
            )

            Tenet.@reindex! outputs(mps) => inputs(mpo)

            @test issetequal([inds(mps; at=i) for i in outputs(mps)], [inds(mpo; at=i) for i in inputs(mpo)])

            # test that the both inputs/outputs appear on the corresponding tensor
            @test all(Site(1:3)) do i
                inds(mps; at=i) ∈ inds(tensors(mpo; at=i))
            end

            @test all(Site(1:3)) do i
                (inds(mpo; at=i), inds(mpo; at=i')) ⊆ inds(tensors(mpo; at=i))
            end
        end

        @testset "regular indices" begin
            # mps-like tensor network
            mps = Quantum(
                TensorNetwork(
                    Tensor[
                        Tensor(rand(2, 2), [:A, :B]), Tensor(rand(2, 2, 2), [:C, :B, :D]), Tensor(rand(2, 2), [:E, :D])
                    ],
                ),
                Dict(site"1" => :A, site"2" => :C, site"3" => :E),
            )

            # mpo-like tensor network
            mpo = Quantum(
                TensorNetwork(
                    Tensor[
                        Tensor(rand(2, 2, 2), [:A, :B, :C]),
                        Tensor(rand(2, 2, 2, 2), [:D, :E, :C, :F]),
                        Tensor(rand(2, 2, 2), [:F, :G, :H]),
                    ],
                ),
                Dict(site"1" => :A, site"1'" => :B, site"2" => :D, site"2'" => :E, site"3" => :G, site"3'" => :H),
            )

            Tenet.@reindex! outputs(mps) => inputs(mpo)

            @test issetequal([inds(mps; at=i) for i in outputs(mps)], [inds(mpo; at=i) for i in inputs(mpo)])

            # test that the both inputs/outputs appear on the corresponding tensor
            @test all(Site(1:3)) do i
                inds(mps; at=i) ∈ inds(tensors(mpo; at=i))
            end

            @test all(Site(1:3)) do i
                (inds(mpo; at=i), inds(mpo; at=i')) ⊆ inds(tensors(mpo; at=i))
            end
        end
    end
end

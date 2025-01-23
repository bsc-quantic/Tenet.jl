@testset "Quantum" begin
    using Tenet: nsites, State, Operator, Scalar

    _tensors = Tensor[Tensor(zeros(2), [:i])]
    tn = TensorNetwork(_tensors)
    qtn = Quantum(tn, Dict(site"1" => :i))
    @test nsites(qtn; set=:inputs) == 0
    @test nsites(qtn; set=:outputs) == 1
    @test issetequal(sites(qtn), [site"1"])
    @test socket(qtn) == State(; dual=false)
    @test inds(qtn; at=site"1") == :i
    @test issetequal(inds(qtn; set=:physical), [:i])
    @test isempty(inds(qtn; set=:virtual))

    # forwarded methods to `TensorNetwork`
    @test TensorNetwork(qtn) == tn
    @test tensors(qtn) == _tensors

    _tensors = Tensor[Tensor(zeros(2), [:i])]
    tn = TensorNetwork(_tensors)
    qtn = Quantum(tn, Dict(site"1'" => :i))
    @test nsites(qtn; set=:inputs) == 1
    @test nsites(qtn; set=:outputs) == 0
    @test issetequal(sites(qtn), [site"1'"])
    @test socket(qtn) == State(; dual=true)
    @test inds(qtn; at=site"1'") == :i
    @test issetequal(inds(qtn; set=:physical), [:i])
    @test isempty(inds(qtn; set=:virtual))

    _tensors = Tensor[Tensor(zeros(2, 2), [:i, :j])]
    tn = TensorNetwork(_tensors)
    qtn = Quantum(tn, Dict(site"1" => :i, site"1'" => :j))
    @test nsites(qtn; set=:inputs) == 1
    @test nsites(qtn; set=:outputs) == 1
    @test issetequal(sites(qtn), [site"1", site"1'"])
    @test socket(qtn) == Operator()
    @test inds(qtn; at=site"1") == :i
    @test inds(qtn; at=site"1'") == :j
    @test issetequal(inds(qtn; set=:physical), [:i, :j])
    @test isempty(inds(qtn; set=:virtual))

    _tensors = Tensor[Tensor(fill(0))]
    tn = TensorNetwork(_tensors)
    qtn = Quantum(tn, Dict())
    @test nsites(qtn; set=:inputs) == 0
    @test nsites(qtn; set=:outputs) == 0
    @test isempty(sites(qtn))
    @test socket(qtn) == Scalar()
    @test isempty(inds(qtn; set=:physical))
    @test isempty(inds(qtn; set=:virtual))

    # detect errors
    _tensors = Tensor[Tensor(zeros(2), [:i]), Tensor(zeros(2), [:i])]
    tn = TensorNetwork(_tensors)
    @test_throws ErrorException Quantum(tn, Dict(site"1" => :j))
    @test_throws ErrorException Quantum(tn, Dict(site"1" => :i))

    @testset "Base.adjoint" begin
        _tensors = Tensor[
            Tensor(rand(ComplexF64, 2, 4, 2), [:i, :link, :j]), Tensor(rand(ComplexF64, 2, 4, 2), [:k, :link, :l])
        ]
        tn = TensorNetwork(_tensors)
        qtn = Quantum(tn, Dict(site"1" => :i, site"2" => :k, site"1'" => :j, site"2'" => :l))

        adjoint_qtn = adjoint(qtn)

        @test nsites(adjoint_qtn; set=:inputs) == nsites(adjoint_qtn; set=:outputs) == 2
        @test issetequal(sites(adjoint_qtn), [site"1", site"2", site"1'", site"2'"])
        @test socket(adjoint_qtn) == Operator()
        @test inds(adjoint_qtn; at=site"1'") == :i # now the indices are flipped
        @test inds(adjoint_qtn; at=site"1") == :j
        @test inds(adjoint_qtn; at=site"2'") == :k
        @test inds(adjoint_qtn; at=site"2") == :l
        @test isapprox(tensors(adjoint_qtn), replace.(conj.(_tensors), :link => Symbol(:link, "'")))
    end

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

            @test issetequal(
                [inds(mps; at=i) for i in sites(mps; set=:outputs)], [inds(mpo; at=i) for i in sites(mpo; set=:inputs)]
            )

            # test that the both inputs/outputs appear on the corresponding tensor
            @test all(Site.(1:3)) do i
                inds(mps; at=i) ∈ inds(tensors(mpo; at=i))
            end

            @test all(Site.(1:3)) do i
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

            @test issetequal(
                [inds(mps; at=i) for i in sites(mps; set=:outputs)], [inds(mpo; at=i) for i in sites(mpo; set=:inputs)]
            )

            # test that the both inputs/outputs appear on the corresponding tensor
            @test all(Site.(1:3)) do i
                inds(mps; at=i) ∈ inds(tensors(mpo; at=i))
            end

            @test all(Site.(1:3)) do i
                (inds(mpo; at=i), inds(mpo; at=i')) ⊆ inds(tensors(mpo; at=i))
            end
        end

        @testset "mps greater than mpo" begin
            # MPS-like tensor network with 4 sites
            mps4sites = Quantum(
                TensorNetwork(
                    Tensor[
                        Tensor(rand(2, 2), [:i, :j]),
                        Tensor(rand(2, 2, 2), [:j, :k, :l]),
                        Tensor(rand(2, 2, 2), [:l, :m, :n]),
                        Tensor(rand(2, 2), [:n, :o]),
                    ],
                ),
                Dict(site"1" => :i, site"2" => :k, site"3" => :m, site"4" => :o),
            )

            # MPO-like tensor network with 3 sites
            mpo3sites = Quantum(
                TensorNetwork(
                    Tensor[
                        Tensor(rand(2, 2, 2), [:i, :j, :k]),
                        Tensor(rand(2, 2, 2, 2), [:l, :m, :k, :n]),
                        Tensor(rand(2, 2, 2), [:o, :p, :n]),
                    ],
                ),
                Dict(site"1" => :i, site"1'" => :j, site"2" => :l, site"2'" => :m, site"3" => :o, site"3'" => :p),
            )

            Tenet.@reindex! outputs(mps4sites) => inputs(mpo3sites)

            mps4inds = [inds(mps4sites; at=i) for i in sites(mps4sites; set=:outputs)]
            mpo3inds = [inds(mpo3sites; at=i) for i in sites(mpo3sites; set=:inputs)]
            # test MPO sites indices are a subset of MPS's but not the opposite
            @test mpo3inds ⊆ mps4inds
            @test !(mps4inds ⊆ mpo3inds)
        end

        @testset "mpo greater than mps" begin
            # MPS-like tensor network with 3 sites
            mps3sites = Quantum(
                TensorNetwork(
                    Tensor[
                        Tensor(rand(2, 2), [:i, :j]), Tensor(rand(2, 2, 2), [:j, :k, :l]), Tensor(rand(2, 2), [:l, :m])
                    ],
                ),
                Dict(site"1" => :i, site"2" => :k, site"3" => :m),
            )

            # MPO-like tensor network with 4 sites
            mpo4sites = Quantum(
                TensorNetwork(
                    Tensor[
                        Tensor(rand(2, 2, 2), [:i, :j, :k]),
                        Tensor(rand(2, 2, 2, 2), [:l, :m, :k, :n]),
                        Tensor(rand(2, 2, 2, 2), [:o, :p, :n, :q]),
                        Tensor(rand(2, 2, 2), [:r, :s, :q]),
                    ],
                ),
                Dict(
                    site"1" => :i,
                    site"1'" => :j,
                    site"2" => :l,
                    site"2'" => :m,
                    site"3" => :o,
                    site"3'" => :p,
                    site"4" => :r,
                    site"4'" => :s,
                ),
            )

            Tenet.@reindex! outputs(mps3sites) => inputs(mpo4sites)

            mps3inds = [inds(mps3sites; at=i) for i in sites(mps3sites; set=:outputs)]
            mpo4inds = [inds(mpo4sites; at=i) for i in sites(mpo4sites; set=:inputs)]
            # test MPS sites indices are a subset of MPO's but not the opposite
            @test mps3inds ⊆ mpo4inds
            @test !(mpo4inds ⊆ mps3inds)
        end
    end
end

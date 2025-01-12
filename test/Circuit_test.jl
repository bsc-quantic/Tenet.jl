@testset "Circuit" begin
    # test empty circuit
    @testset let circuit = Circuit()
        @test TensorNetwork(circuit) == TensorNetwork()
        @test isempty(sites(circuit))
        @test isempty(lanes(circuit))
        @test isempty(inds(circuit; set=:inputs))
        @test isempty(inds(circuit; set=:outputs))
        @test isempty(inds(circuit; set=:physical))
        @test isempty(inds(circuit; set=:virtual))
    end

    # test applying a single lane gate
    @testset let circuit = Circuit(), gate = Gate(zeros(2, 2), [site"1'", site"1"])
        push!(circuit, gate)
        @test issetequal(sites(circuit), [site"1'", site"1"])
        @test issetequal(lanes(circuit), [Lane(1)])
        @test issetequal(sites(circuit; set=:inputs), [site"1'"])
        @test issetequal(sites(circuit; set=:outputs), [site"1"])
        @test Tenet.ninds(circuit) == 2
        @test Tenet.ninds(circuit; set=:inputs) == 1
        @test Tenet.ninds(circuit; set=:outputs) == 1
        @test Tenet.ninds(circuit; set=:physical) == 2
        @test Tenet.ninds(circuit; set=:virtual) == 0
        @test Tenet.ntensors(circuit) == 1
        @test parent(tensors(circuit; at=site"1")) == parent(Tensor(gate))
        @test Tenet.moment(circuit, site"1'") == Moment(Lane(1), 1)
        @test Tenet.moment(circuit, site"1") == Moment(Lane(1), 2)
    end

    # test applying a gate with multiple lanes
    @testset let circuit = Circuit(), gate = Gate(zeros(2, 2, 2, 2), [site"1'", site"2'", site"1", site"2"])
        push!(circuit, gate)
        @test issetequal(sites(circuit), [site"1'", site"1"])
        @test issetequal(lanes(circuit), [Lane(1), Lane(2)])
        @test issetequal(sites(circuit; set=:inputs), [site"1'", site"2'"])
        @test issetequal(sites(circuit; set=:outputs), [site"1", site"2"])
        @test Tenet.ninds(circuit) == 4
        @test Tenet.ninds(circuit; set=:inputs) == 2
        @test Tenet.ninds(circuit; set=:outputs) == 2
        @test Tenet.ninds(circuit; set=:physical) == 4
        @test Tenet.ninds(circuit; set=:virtual) == 0
        @test Tenet.ntensors(circuit) == 1
        @test parent(tensors(circuit; at=site"1")) == parent(Tensor(gate))
        @test Tenet.moment(circuit, site"1'") == Moment(Lane(1), 1)
        @test Tenet.moment(circuit, site"2'") == Moment(Lane(2), 1)
        @test Tenet.moment(circuit, site"1") == Moment(Lane(1), 2)
        @test Tenet.moment(circuit, site"2") == Moment(Lane(2), 2)
    end

    # test applying two gates in the same lane
    @testset let circuit = Circuit(),
        gate1 = Gate(zeros(2, 2), [site"1'", site"1"]),
        gate2 = Gate(ones(2, 2), [site"1'", site"1"])

        push!(circuit, gate1)
        push!(circuit, gate2)
        @test issetequal(sites(circuit), [site"1'", site"1"])
        @test issetequal(lanes(circuit), [Lane(1)])
        @test issetequal(sites(circuit; set=:inputs), [site"1'"])
        @test issetequal(sites(circuit; set=:outputs), [site"1"])
        @test Tenet.ninds(circuit) == 3
        @test Tenet.ninds(circuit; set=:inputs) == 1
        @test Tenet.ninds(circuit; set=:outputs) == 1
        @test Tenet.ninds(circuit; set=:physical) == 2
        @test Tenet.ninds(circuit; set=:virtual) == 1
        @test Tenet.ntensors(circuit) == 2
        @test parent(tensors(circuit; at=site"1'")) == parent(Tensor(gate1))
        @test parent(tensors(circuit; at=site"1")) == parent(Tensor(gate2))
        @test Tenet.moment(circuit, site"1'") == Moment(Lane(1), 1)
        @test Tenet.moment(circuit, site"1") == Moment(Lane(1), 3)
    end

    # test applying the same gate twice
    @testset let circuit = Circuit(), gate = Gate(zeros(2, 2), [site"1'", site"1"])
        push!(circuit, gate)
        push!(circuit, gate)
        @test issetequal(sites(circuit), [site"1'", site"1"])
        @test issetequal(lanes(circuit), [Lane(1)])
        @test issetequal(sites(circuit; set=:inputs), [site"1'"])
        @test issetequal(sites(circuit; set=:outputs), [site"1"])
        @test Tenet.ninds(circuit) == 3
        @test Tenet.ninds(circuit; set=:inputs) == 1
        @test Tenet.ninds(circuit; set=:outputs) == 1
        @test Tenet.ninds(circuit; set=:physical) == 2
        @test Tenet.ninds(circuit; set=:virtual) == 1
        @test Tenet.ntensors(circuit) == 2
        @test parent(tensors(circuit; at=site"1'")) == parent(Tensor(gate))
        @test parent(tensors(circuit; at=site"1")) == parent(Tensor(gate))
        @test Tenet.moment(circuit, site"1'") == Moment(Lane(1), 1)
        @test Tenet.moment(circuit, site"1") == Moment(Lane(1), 3)
    end

    # test applying two gates in different lanes
    @testset let circuit = Circuit(),
        gate1 = Gate(zeros(2, 2), [site"1'", site"1"]),
        gate2 = Gate(ones(2, 2), [site"2'", site"2"])

        push!(circuit, gate1)
        push!(circuit, gate2)
        @test issetequal(sites(circuit), [site"1'", site"1", site"2'", site"2"])
        @test issetequal(lanes(circuit), [Lane(1), Lane(2)])
        @test issetequal(sites(circuit; set=:inputs), [site"1'", site"2'"])
        @test issetequal(sites(circuit; set=:outputs), [site"1", site"2"])
        @test Tenet.ninds(circuit) == 4
        @test Tenet.ninds(circuit; set=:inputs) == 2
        @test Tenet.ninds(circuit; set=:outputs) == 2
        @test Tenet.ninds(circuit; set=:physical) == 4
        @test Tenet.ninds(circuit; set=:virtual) == 0
        @test Tenet.ntensors(circuit) == 2
        @test parent(tensors(circuit; at=site"1'")) == parent(Tensor(gate1))
        @test parent(tensors(circuit; at=site"1")) == parent(Tensor(gate1))
        @test parent(tensors(circuit; at=site"2'")) == parent(Tensor(gate2))
        @test parent(tensors(circuit; at=site"2")) == parent(Tensor(gate2))
        @test Tenet.moment(circuit, site"1'") == Moment(Lane(1), 1)
        @test Tenet.moment(circuit, site"1") == Moment(Lane(1), 2)
        @test Tenet.moment(circuit, site"2'") == Moment(Lane(2), 1)
        @test Tenet.moment(circuit, site"2") == Moment(Lane(2), 2)
    end

    # test applying two gates with a shared lane
    @testset let circuit = Circuit(),
        gate1 = Gate(zeros(2, 2), [site"1'", site"1"]),
        gate2 = Gate(ones(2, 2, 2, 2), [site"1'", site"2'", site"1", site"2"])

        push!(circuit, gate1)
        push!(circuit, gate2)
        @test issetequal(sites(circuit), [site"1'", site"1", site"2'", site"2"])
        @test issetequal(lanes(circuit), [Lane(1), Lane(2)])
        @test issetequal(sites(circuit; set=:inputs), [site"1'", site"2'"])
        @test issetequal(sites(circuit; set=:outputs), [site"1", site"2"])
        @test Tenet.ninds(circuit) == 5
        @test Tenet.ninds(circuit; set=:inputs) == 2
        @test Tenet.ninds(circuit; set=:outputs) == 2
        @test Tenet.ninds(circuit; set=:physical) == 4
        @test Tenet.ninds(circuit; set=:virtual) == 1
        @test Tenet.ntensors(circuit) == 2
        @test parent(tensors(circuit; at=site"1'")) == parent(Tensor(gate1))
        @test parent(tensors(circuit; at=site"1")) == parent(Tensor(gate2))
        @test parent(tensors(circuit; at=site"2'")) == parent(Tensor(gate2))
        @test parent(tensors(circuit; at=site"2")) == parent(Tensor(gate2))
        @test Tenet.moment(circuit, site"1'") == Moment(Lane(1), 1)
        @test Tenet.moment(circuit, site"1") == Moment(Lane(1), 3)
        @test Tenet.moment(circuit, site"2'") == Moment(Lane(2), 1)
        @test Tenet.moment(circuit, site"2") == Moment(Lane(2), 2)
    end
end

using Test
using Tenet
using Tenet: Moment, nsites

@testset "empty circuit" begin
    circuit = Circuit()
    test_tensornetwork(circuit)
    test_pluggable(circuit)

    @test TensorNetwork(circuit) == TensorNetwork()
    @test isempty(sites(circuit))
    @test isempty(lanes(circuit))
    @test isempty(inds(circuit; set=:inputs))
    @test isempty(inds(circuit; set=:outputs))
    @test isempty(inds(circuit; set=:physical))
    @test isempty(inds(circuit; set=:virtual))
    @test collect(circuit) == Gate[]

    qtn = Quantum(circuit)
    @test issetequal(sites(qtn), sites(circuit))
    @test issetequal(tensors(qtn), tensors(circuit))
end

@testset "apply single-lane gate" begin
    circuit = Circuit()
    gate = Gate(zeros(2, 2), [site"1'", site"1"])
    push!(circuit, gate)

    test_tensornetwork(circuit)
    test_pluggable(circuit)

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
    @test parent.(Tensor.(circuit)) == parent.(Tensor.([gate]))
    @test sites.(circuit) == sites.([gate])

    qtn = Quantum(circuit)
    @test issetequal(sites(qtn), sites(circuit))
    @test issetequal(tensors(qtn), tensors(circuit))
end

@testset "apply multi-lane gate" begin
    circuit = Circuit()
    gate = Gate(zeros(2, 2, 2, 2), [site"1'", site"2'", site"1", site"2"])
    push!(circuit, gate)

    test_tensornetwork(circuit)
    test_pluggable(circuit)

    @test issetequal(sites(circuit), [site"1'", site"1", site"2'", site"2"])
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
    @test parent.(Tensor.(circuit)) == parent.(Tensor.([gate]))
    @test sites.(circuit) == sites.([gate])

    qtn = Quantum(circuit)
    @test issetequal(sites(qtn), sites(circuit))
    @test issetequal(tensors(qtn), tensors(circuit))
end

@testset "apply two gates in the same lane" begin
    circuit = Circuit()
    gate1 = Gate(zeros(2, 2), [site"1'", site"1"])
    gate2 = Gate(ones(2, 2), [site"1'", site"1"])

    push!(circuit, gate1)
    push!(circuit, gate2)

    test_tensornetwork(circuit)
    test_pluggable(circuit)

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
    @test parent.(Tensor.(circuit)) == parent.(Tensor.([gate1, gate2]))
    @test sites.(circuit) == sites.([gate1, gate2])

    qtn = Quantum(circuit)
    @test issetequal(sites(qtn), sites(circuit))
    @test issetequal(tensors(qtn), tensors(circuit))
end

@testset "apply the same gate twice" begin
    circuit = Circuit()
    gate = Gate(zeros(2, 2), [site"1'", site"1"])
    push!(circuit, gate)
    push!(circuit, gate)

    test_tensornetwork(circuit)
    test_pluggable(circuit)

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
    @test parent.(Tensor.(circuit)) == parent.(Tensor.([gate, gate]))
    @test sites.(circuit) == sites.([gate, gate])

    qtn = Quantum(circuit)
    @test issetequal(sites(qtn), sites(circuit))
    @test issetequal(tensors(qtn), tensors(circuit))
end

# test applying two gates in different lanes
@testset "apply two gates in different lanes" begin
    circuit = Circuit()
    gate1 = Gate(zeros(2, 2), [site"1'", site"1"])
    gate2 = Gate(ones(2, 2), [site"2'", site"2"])

    push!(circuit, gate1)
    push!(circuit, gate2)

    test_tensornetwork(circuit)
    test_pluggable(circuit)

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
    @test parent.(Tensor.(circuit)) == parent.(Tensor.([gate1, gate2]))
    @test sites.(circuit) == sites.([gate1, gate2])

    qtn = Quantum(circuit)
    @test issetequal(sites(qtn), sites(circuit))
    @test issetequal(tensors(qtn), tensors(circuit))
end

@testset "apply two gates with a shared lane" begin
    circuit = Circuit()
    gate1 = Gate(zeros(2, 2), [site"1'", site"1"])
    gate2 = Gate(ones(2, 2, 2, 2), [site"1'", site"2'", site"1", site"2"])

    push!(circuit, gate1)
    push!(circuit, gate2)

    test_tensornetwork(circuit)
    test_pluggable(circuit)

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
    @test parent.(Tensor.(circuit)) == parent.(Tensor.([gate1, gate2]))
    @test sites.(circuit) == sites.([gate1, gate2])

    qtn = Quantum(circuit)
    @test issetequal(sites(qtn), sites(circuit))
    @test issetequal(tensors(qtn), tensors(circuit))
end

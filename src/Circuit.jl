using BijectiveDicts: BijectiveIdDict

struct Gate
    tensor::Tensor
    sites::Vector{Site}

    function Gate(tensor::Tensor, sites)
        @assert ndims(tensor) == length(sites)
        @assert allunique(sites)
        @assert allunique(inds(tensor))
        new(tensor, sites)
    end
end

Gate(array::AbstractArray, sites) = Gate(Tensor(array, [gensym(:i) for _ in 1:ndims(array)]), sites)

Tensor(gate::Gate) = gate.tensor
Base.parent(gate::Gate) = Tensor(gate)
inds(gate::Gate) = inds(Tensor(gate))

sites(gate::Gate; kwargs...) = sites(sort_nt(values(kwargs)), gate)
sites(::@NamedTuple{}, gate::Gate) = Tuple(gate.sites)

function sites(kwargs::@NamedTuple{set::Symbol}, gate::Gate)
    pred = if kwargs.set === :outputs
        !isdual
    elseif kwargs.set === :inputs
        isdual
    else
        throw(ArgumentError("Expected set to be one of `:inputs` or `:outputs`, but got $(kwargs.set)"))
    end
    return filter(pred, sites(gate))
end

lanes(gate::Gate) = unique(Iterators.map(Tenet.Lane, sites(gate)))

Base.replace(gate::Gate, old_new::Pair{Symbol,Symbol}...) = Gate(replace(Tensor(gate), old_new...), sites(gate))

function Base.replace(gate::Gate, old_new::Pair{Site,Symbol}...)
    mapping = ImmutableDict(Pair.(sites(gate), inds(gate)))
    old_new = map(old_new) do (old, new)
        mapping[old] => new
    end
    return replace(gate, old_new...)
end

resetindex(gate::Gate) = replace(gate, [ind => gensym(:i) for ind in inds(gate)]...)

struct Circuit <: AbstractQuantum
    tn::TensorNetwork
    moments::BijectiveIdDict{Moment,Symbol} # mapping between indices and `Moment`s
    inputs::Dict{Lane,Int}
    outputs::Dict{Lane,Int} # current moment for each lane
end

Circuit() = Circuit(TensorNetwork(), BijectiveIdDict{Symbol,Moment}(), Dict(), Dict())

TensorNetwork(circuit::Circuit) = circuit.tn
# TODO conversion between `Quantum and `Circuit`

function moment(circuit::Circuit, site::Site)
    lane = Lane(site)
    t = isdual(site) ? circuit.inputs[lane] : circuit.outputs[lane]
    return Moment(lane, t)
end

inds(kwargs::@NamedTuple{at::Site}, circuit::Circuit) = circuit.moments[moment(circuit, kwargs.at)]

# NOTE `tensors(; at)` is implemented in `Quantum.jl` for `AbstractQuantum`

function sites(::@NamedTuple{}, circuit::Circuit)
    keys(circuit.inputs) ∪ keys(circuit.outputs)
end

function sites(::@NamedTuple{set::Symbol}, circuit::Circuit)
    if set === :inputs
        return Site.(keys(circuit.inputs); dual=true)
    elseif set === :outputs
        return Site.(keys(circuit.outputs))
    else
        throw(ArgumentError("Expected set to be one of `:inputs` or `:outputs`, but got $(set)"))
    end
end

# NOTE `lanes` is implemented in `Quantum.jl` for `AbstractQuantum`

function Base.push!(circuit::Circuit, gate::Gate)
    # only gates with same input/output lanes are allowed, even if `Circuit` could support unbalanced inputs/outputs
    @assert issetequal(sites(gate; set=:outputs), adjoint.(sites(gate; set=:inputs)))

    connecting_lanes = lanes(gate) ∩ Lane.(sites(circuit; set=:outputs))
    new_lanes = setdiff(lanes(gate), connecting_lanes)

    # reindex gate to match circuit indices
    gate = replace(gate, [site' => inds(circuit; at=site) for site in Iterators.map(Site, connecting_lanes)])

    # add gate to circuit
    push!(TensorNetwork(circuit), Tensor(gate))

    # update moments: point to new output indices
    for lane in new_lanes
        circuit.moments[Moment(lane, 1)] = inds(gate; at=Site(lane; dual=true))
        circuit.moments[Moment(lane, 2)] = inds(gate; at=Site(lane))
        circuit.inputs[lane] = 1
        circuit.outputs[lane] = 2
    end

    for lane in connecting_lanes
        circuit.outputs[lane] += 1
        t = circuit.outputs[lane]
        moment = Moment(lane, t)
        circuit.moments[moment] = inds(gate; at=Site(lane))
    end

    return circuit
end

# TODO iterative walk through the gates

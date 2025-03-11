using BijectiveDicts: BijectiveDict

struct Circuit <: AbstractQuantum
    tn::TensorNetwork
    moments::BijectiveDict{Moment,Symbol} # mapping between indices and `Moment`s
    inputs::Dict{Lane,Int}
    outputs::Dict{Lane,Int} # current moment for each lane
    ordered_gates::Vector{Gate} # used in iterate
end

function Circuit()
    Circuit(TensorNetwork(), BijectiveDict(Dict{Moment,Symbol}(), Dict{Symbol,Moment}()), Dict(), Dict(), Gate[])
end

TensorNetwork(circuit::Circuit) = circuit.tn

# TODO conversion from `Quantum to `Circuit`
function Quantum(circuit::Circuit)
    Quantum(TensorNetwork(circuit), Dict([site => inds(circuit; at=site) for site in sites(circuit)]))
end

moments(circuit::Circuit) = collect(keys(circuit.moments))

function moment(circuit::Circuit, site::Site)
    lane = Lane(site)
    t = isdual(site) ? circuit.inputs[lane] : circuit.outputs[lane]
    return Moment(lane, t)
end

inds(kwargs::@NamedTuple{at::S}, circuit::Circuit) where {S<:Site} = circuit.moments[moment(circuit, kwargs.at)]
inds(kwargs::@NamedTuple{at::Moment}, circuit::Circuit) = circuit.moments[kwargs.at]

# NOTE `inds(; set)` is implemented in `Quantum.jl` for `AbstractQuantum`, but reimplemented here for performance
function inds(kwargs::@NamedTuple{set::Symbol}, circuit::Circuit)
    if kwargs.set ∈ (:inputs, :outputs)
        return [inds(circuit; at=site) for site in sites(circuit; kwargs...)]
    elseif kwargs.set === :physical
        return [inds(circuit; at=site) for site in sites(circuit)]
    elseif kwargs.set === :virtual
        return setdiff(inds(circuit), inds(circuit; set=:physical))
    else
        return inds(TensorNetwork(circuit); set=kwargs.set)
    end
end

# NOTE `tensors(; at)` is implemented in `Quantum.jl` for `AbstractQuantum`, but reimplemented for performance
function tensors(kwargs::@NamedTuple{at::S}, circuit::Circuit) where {S<:Site}
    only(tensors(circuit; contains=inds(circuit; kwargs...)))
end

# NOTE `tensors(; at::Moment)` not fully defined

sites(::@NamedTuple{}, circuit::Circuit) = vcat(sites(circuit; set=:inputs), sites(circuit; set=:outputs))

function sites(kwargs::@NamedTuple{set::Symbol}, circuit::Circuit)
    if kwargs.set === :all
        sites(circuit)
    elseif kwargs.set === :inputs
        return Site[Site.(keys(circuit.inputs); dual=true)...]
    elseif kwargs.set === :outputs
        return Site[Site.(keys(circuit.outputs))...]
    else
        throw(ArgumentError("Expected set to be one of `:inputs` or `:outputs`, but got $(kwargs.set)"))
    end
end

# NOTE `lanes` is implemented in `Quantum.jl` for `AbstractQuantum`

function Base.push!(circuit::Circuit, gate::Gate)
    # only gates with same input/output lanes are allowed, even if `Circuit` could support unbalanced inputs/outputs
    @assert issetequal(sites(gate; set=:outputs), adjoint.(sites(gate; set=:inputs)))

    connecting_lanes = lanes(gate) ∩ Lane.(sites(circuit; set=:outputs))
    new_lanes = setdiff(lanes(gate), connecting_lanes)

    # reindex gate to match circuit indices
    gate = resetinds(gate)
    if !isempty(connecting_lanes)
        gate = replace(gate, [site' => inds(circuit; at=site) for site in Iterators.map(Site, connecting_lanes)]...)
    end

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

    push!(circuit.ordered_gates, gate)

    return circuit
end

Base.eltype(::Type{Circuit}) = Gate
Base.IteratorSize(::Type{Circuit}) = Base.HasLength()

# TODO choose between breadth-first and depth-first traversing algorithms
"""
    Base.iterate(circuit::Circuit[, state=1])

Iterate over the gates in `circuit` in the order they were added.
"""
Base.iterate(circuit::Circuit, state=1) = iterate(circuit.ordered_gates, state)

# NOTE if not specialized, it will fallback to the `TensorNetwork` method with returns `Vector{Tensor}`
Base.collect(circuit::Circuit) = copy(circuit.ordered_gates)

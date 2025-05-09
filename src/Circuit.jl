using BijectiveDicts: BijectiveDict

"""
    Gate

A `Gate` is a [`Tensor`](@ref) together with a set of [`Site`](@ref)s that represent the input/output indices.
It is similar to the relation between [`Quantum`](@ref) and [`TensorNetwork`](@ref), but only applies to one tensor.

Although it is not a [`AbstractQuantum`](@ref), it can be converted to [`TensorNetwork`](@ref) or [`Quantum`](@ref).
"""
struct Gate
    tensor::Tensor
    sites::Vector{Site}

    function Gate(tensor::Tensor, sites)
        @assert ndims(tensor) == length(sites)
        @assert allunique(sites)
        @assert allunique(inds(tensor))
        new(tensor, collect(sites))
    end
end

Gate(array::AbstractArray, sites) = Gate(Tensor(array, [gensym(:i) for _ in 1:ndims(array)]), sites)
Base.copy(gate::Gate) = Gate(copy(Tensor(gate)), sites(gate))

Tensor(gate::Gate) = gate.tensor
Base.parent(gate::Gate) = Tensor(gate)

TensorNetwork(gate::Gate) = TensorNetwork([Tensor(gate)])
Quantum(gate::Gate) = Quantum(TensorNetwork(gate), Dict(sites(gate) .=> inds(gate)))

# AbstractTensorNetwork methods
inds(gate::Gate; kwargs...) = inds(sort_nt(values(kwargs)), gate)
inds(::@NamedTuple{}, gate::Gate) = inds(Tensor(gate))
inds(kwargs::@NamedTuple{at::S}, gate::Gate) where {S<:Site} = inds(gate)[findfirst(isequal(kwargs.at), sites(gate))]
function inds(kwargs::@NamedTuple{set::Symbol}, gate::Gate)
    if kwargs.set ∈ (:all, :open, :physical)
        return inds(gate)
    elseif kwargs.set ∈ (:inner, :hyper, :virtual)
        return Symbol[]
    elseif kwargs.set === :inputs
        return last.(
            Iterators.filter(zip(sites(gate), inds(gate))) do (site, ind)
                isdual(site)
            end,
        )
    elseif kwargs.set === :outputs
        return last.(
            Iterators.filter(zip(sites(gate), inds(gate))) do (site, ind)
                !isdual(site)
            end,
        )
    else
        error(
            "Expected set to be one of `:all`, `:open`, `:physical`, `:inner`, `:hyper`, `:virtual`, or `:inputs`, but got $(kwargs.set)",
        )
    end
end

tensors(gate::Gate; kwargs...) = tensors(sort_nt(values(kwargs)), gate)
tensors(::@NamedTuple{}, gate::Gate) = Tensor[Tensor(gate)]

# AbstractQuantum methods
sites(gate::Gate; kwargs...) = sites(sort_nt(values(kwargs)), gate)
sites(::@NamedTuple{}, gate::Gate) = Tuple(gate.sites)

function sites(kwargs::@NamedTuple{set::Symbol}, gate::Gate)
    pred = if kwargs.set === :all
        _ -> true
    elseif kwargs.set === :outputs
        !isdual
    elseif kwargs.set === :inputs
        isdual
    else
        throw(ArgumentError("Expected set to be one of `:inputs` or `:outputs`, but got $(kwargs.set)"))
    end
    return filter(pred, sites(gate))
end

function sites(kwargs::@NamedTuple{at::Symbol}, gate::Gate)
    loc = findfirst(==(kwargs.at), inds(gate))
    if isnothing(log)
        throw(ArgumentError("Index $(kwargs.at) not found in $(inds(gate))"))
    end

    return sites(gate)[loc]
end

nsites(gate::Gate; kwargs...) = length(sites(gate; kwargs...))

nlanes(gate::Gate) = length(lanes(gate))
lanes(gate::Gate) = unique(Iterators.map(Tenet.Lane, sites(gate)))

socket(::Gate) = Operator()

Base.:(==)(a::Gate, b::Gate) = sites(a) == sites(b) && Tensor(a) == Tensor(b)

Base.replace(gate::Gate) = gate
Base.replace(gate::Gate, old_new::Pair{Symbol,Symbol}...) = Gate(replace(Tensor(gate), old_new...), sites(gate))

function Base.replace(gate::Gate, old_new::Pair{<:Site,Symbol}...)
    mapping = Base.ImmutableDict(Pair.(sites(gate), inds(gate))...)
    old_new = map(old_new) do (old, new)
        mapping[old] => new
    end
    return replace(gate, old_new...)
end

resetinds(gate::Gate; init=nothing) = replace(gate, [ind => gensym(:i) for ind in inds(gate)]...)

function Base.merge!(qtn::AbstractQuantum, gate::Gate; reset=false)
    @assert isconnectable(qtn, gate)
    merge!(qtn, Quantum(gate); reset)
    return qtn
end

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

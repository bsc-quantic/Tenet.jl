using ValSplit

"""
    Gate

A `Gate` is a [`Tensor`](@ref) together with a set of [`Site`](@ref)s that represent the input/output indices; i.e.
fulfills the Pluggable interface.
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

################################################################################
# TODO remove when `Quantum` is removed
TensorNetwork(gate::Gate) = TensorNetwork([Tensor(gate)])
Quantum(gate::Gate) = Quantum(TensorNetwork(gate), Dict(sites(gate) .=> inds(gate)))

# function Base.merge!(qtn::AbstractQuantum, gate::Gate; reset=false)
#     @assert isconnectable(qtn, gate)
#     merge!(qtn, Quantum(gate); reset)
#     return qtn
# end
################################################################################

@propagate_inbounds Base.getindex(gate::Gate, key::Vararg{<:Integer}) = getindex(parent(gate), key...)

# TODO should we just compare the arrays with the appropiate permutation?
Base.:(==)(a::Gate, b::Gate) = sites(a) == sites(b) && Tensor(a) == Tensor(b)

# Tensor Network methods
inds(gate::Gate; kwargs...) = inds(sort_nt(values(kwargs)), gate)
inds(::@NamedTuple{}, gate::Gate) = inds(Tensor(gate))

function extract_kwarg_set(::Val{kwargs}) where {kwargs}
    @assert kwargs isa @NamedTuple{set::Symbol}
    return kwargs.set
end

@valsplit function inds(Val(kwargs::@NamedTuple{set::Symbol}), gate::Gate)
    error(
        "Expected set to be one of `:all`, `:open`, `:physical`, `:inner`, `:hyper`, `:virtual`, or `:inputs`, but got $(extract_kwarg_set(kwargs))",
    )
end

inds(::Val{(; set = :all)}, gate::Gate) = inds(gate)
inds(::Val{(; set = :open)}, gate::Gate) = inds(gate)
inds(::Val{(; set = :physical)}, gate::Gate) = inds(gate)

inds(::Val{(; set = :inner)}, gate::Gate) = Symbol[]
inds(::Val{(; set = :hyper)}, gate::Gate) = Symbol[]
inds(::Val{(; set = :virtual)}, gate::Gate) = Symbol[]

function inds(::Val{(; set = :inputs)}, gate::Gate)
    last.(
        Iterators.filter(zip(sites(gate), inds(gate))) do (site, ind)
            isdual(site)
        end
    )
end

function inds(::Val{(; set = :outputs)}, gate::Gate)
    last.(
        Iterators.filter(zip(sites(gate), inds(gate))) do (site, ind)
            !isdual(site)
        end
    )
end

tensors(gate::Gate; kwargs...) = tensors(sort_nt(values(kwargs)), gate)
tensors(::@NamedTuple{}, gate::Gate) = Tensor[Tensor(gate)]

Base.replace(gate::Gate) = gate
Base.replace(gate::Gate, old_new::Pair{Symbol,Symbol}...) = Gate(replace(Tensor(gate), old_new...), sites(gate))

# Pluggable methods
sites(gate::Gate; kwargs...) = sites(sort_nt(values(kwargs)), gate)
sites(::@NamedTuple{}, gate::Gate) = Tuple(gate.sites)

@valsplit function sites(Val(kwargs::@NamedTuple{set::Symbol}), gate::Gate)
    throw(ArgumentError("Expected plugset to be one of `:inputs` or `:outputs`, but got $(kwargs.plugset)"))
end

sites(::Val{(; set = :all)}, gate::Gate) = sites(gate)
sites(::Val{(; set = :inputs)}, gate::Gate) = filter(isdual, sites(gate))
sites(::Val{(; set = :outputs)}, gate::Gate) = filter(!isdual, sites(gate))

function sites(kwargs::@NamedTuple{at::Symbol}, gate::Gate)
    loc = findfirst(==(kwargs.at), inds(gate))
    if isnothing(log)
        throw(ArgumentError("Index $(kwargs.at) not found in $(inds(gate))"))
    end

    return sites(gate)[loc]
end

nsites(gate::Gate; kwargs...) = length(sites(gate; kwargs...))

inds(kwargs::@NamedTuple{at::S}, gate::Gate) where {S<:Site} = inds(gate)[findfirst(isequal(kwargs.at), sites(gate))]

socket(::Gate) = Operator()

function Base.replace(gate::Gate, old_new::Pair{<:Site,Symbol}...)
    mapping = Base.ImmutableDict(Pair.(sites(gate), inds(gate))...)
    old_new = map(old_new) do (old, new)
        mapping[old] => new
    end
    return replace(gate, old_new...)
end

# Ansatz methods
nlanes(gate::Gate) = length(lanes(gate))
lanes(gate::Gate) = unique(Iterators.map(Tenet.Lane, sites(gate)))

resetinds(gate::Gate; init=nothing) = replace(gate, [ind => gensym(:i) for ind in inds(gate)]...)

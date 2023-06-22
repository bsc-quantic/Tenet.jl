using LinearAlgebra
using UUIDs: uuid4
using ValSplit
using Bijections

"""
    Quantum

Tensor Network [`Ansatz`](@ref) that has a notion of sites and directionality (input/output).
"""
abstract type Quantum <: Ansatz end

# NOTE Storing `Plug` type on type parameters is not compatible with `Composite` ansatz. Use Holy traits instead.
metadata(::Type{Quantum}) = NamedTuple{(:plug, :interlayer),Tuple{Type{<:Plug},Vector{Bijection{Int,Symbol}}}}

function checkmeta(::Type{Quantum}, tn::TensorNetwork)
    # TODO run this check depending if State or Operator
    length(tn.interlayer) >= 1 || return false

    # meta's indices exist
    all(bij -> values(bij) ⊆ labels(tn), tn.interlayer) || return false

    return true
end

abstract type Boundary end
abstract type Open <: Boundary end
abstract type Periodic <: Boundary end

function boundary end
boundary(::T) where {T<:TensorNetwork} = boundary(T)
boundary(::Type{T}) where {T<:TensorNetwork} = boundary(ansatz(T))

abstract type Plug end
abstract type Property <: Plug end
abstract type State <: Plug end
abstract type Operator <: Plug end

function plug end
plug(tn::TensorNetwork{<:Quantum}) = tn.plug
plug(T::Type{<:TensorNetwork}) = plug(ansatz(T))

sites(tn::TensorNetwork) = collect(mapreduce(keys, ∪, tn.interlayer))

labels(tn::TensorNetwork, ::Val{:plug}) = unique(Iterators.flatten(Iterators.map(values, tn.interlayer)))
labels(tn::TensorNetwork, ::Val{:plug}, site) = last(tn.interlayer)[site] # labels(tn, Val(:in), site) ∪ labels(tn, Val(:out), site)
labels(tn::TensorNetwork, ::Val{:virtual}) = setdiff(labels(tn, Val(:all)), labels(tn, Val(:plug)))

tensors(tn::TensorNetwork{<:Quantum}, site::Integer, args...) = tensors(plug(tn), tn, site, args...)
tensors(::Type{State}, tn::TensorNetwork{<:Quantum}, site) = select(tn, labels(tn, :plug, site)) |> only
@valsplit 4 tensors(T::Type{Operator}, tn::TensorNetwork{<:Quantum}, site, dir::Symbol) =
    throw(MethodError(sites, "dir=$dir not recognized"))

function Base.replace!(tn::TensorNetwork{<:Quantum}, old_new::Pair{Symbol,Symbol})
    # replace indices in tensor network
    Base.@invoke replace!(tn::TensorNetwork, old_new::Pair{Symbol,Symbol})

    old, new = old_new

    # replace indices in interlayers (quantum-specific)
    for interlayer in Iterators.filter(∋(old) ∘ image, tn.interlayer)
        site = interlayer(old)
        delete!(interlayer, site)
        interlayer[site] = new
    end

    return tn
end

## `Composite` type
abstract type Composite{Ts<:Tuple} <: Quantum end
Composite(@nospecialize(Ts::Type{<:Quantum}...)) = Composite{Tuple{Ts...}}
Base.fieldtypes(::Type{Composite{Ts}}) where {Ts} = fieldtypes(Ts)

metadata(A::Type{<:Composite}) = NamedTuple{(:layermeta,),Tuple{Vector{Dict{Symbol,Any}}}}

function checkmeta(As::Type{<:Composite}, tn::TensorNetwork)
    for (i, A) in enumerate(fieldtypes(As))
        tn_view = layers(tn, i)
        checkansatz(tn_view)
    end

    return true
end

Base.length(@nospecialize(T::Type{<:Composite})) = length(fieldtypes(T))

# TODO create view of TN
function layers(tn::TensorNetwork{As}, i) where {As<:Composite}
    A = fieldtypes(As)[i]
    layer_plug = tn.layermeta[i][:plug] # TODO more programmatic access (e.g. plug(tn, i)?)
    meta = tn.layermeta[i]

    if layer_plug <: State && 1 < i < length(fieldtypes(As))
        throw(ErrorException("Layer #$i is a state but it is not a extreme layer"))
    end

    interlayer = if layer_plug <: State
        i == 1 ? [first(tn.interlayer)] : [last(tn.interlayer)]
    elseif layer_plug <: Operator
        # shift if first layer is a state
        plug(first(fieldtypes(As))) <: State && (i = i - 1)
        tn.interlayer[i:i+1]
    end

    return TensorNetwork{A}(
        filter(tensor -> get(tensor.meta, :layer, nothing) == i, tensors(tn));
        plug=layer_plug,
        interlayer,
        meta...
    )
end

Base.merge(::Type{State}, ::Type{State}) = Property
Base.merge(::Type{State}, ::Type{Operator}) = State
Base.merge(::Type{Operator}, ::Type{State}) = State
Base.merge(::Type{Operator}, ::Type{Operator}) = Operator

# TODO implement hcat when QA or QB <: Composite
function Base.hcat(A::TensorNetwork{QA}, B::TensorNetwork{QB}) where {QA<:Quantum,QB<:Quantum}
    issetequal(sites(A), sites(B)) ||
        throw(DimensionMismatch("A and B must contain the same set of sites in order to connect them"))

    # rename connector indices
    newinds = Dict([s => Symbol(uuid4()) for s in sites(A)])

    B = copy(B)

    for site in sites(B)
        a = labels(A, :plug, site)
        b = labels(B, :plug, site)
        if a != b && a ∉ labels(B)
            replace!(B, b => a)
        end
    end

    # rename inner indices of B to avoid hyperindices
    replace!(B, [i => Symbol(uuid4()) for i in labels(B, :inner)]...)

    # TODO refactor this part to be compatible with more layers
    foreach(tensor -> tensor.meta[:layer] = 1, tensors(A))
    foreach(tensor -> tensor.meta[:layer] = 2, tensors(B))

    combined_plug = merge(plug(A), plug(B))

    # merge tensors and indices
    interlayer = [A.interlayer..., collect(Iterators.drop(B.interlayer, 1))...]

    # TODO merge metadata?
    layermeta = Dict{Symbol,Any}[
        Dict(Iterators.filter(((k, v),) -> k !== :interlayer, pairs(A.metadata))),
        Dict(Iterators.filter(((k, v),) -> k !== :interlayer, pairs(B.metadata))),
    ]

    return TensorNetwork{Composite(QA, QB)}([tensors(A)..., tensors(B)...]; plug = combined_plug, interlayer, layermeta)
end

Base.hcat(tns::TensorNetwork...) = reduce(hcat, tns)

function Base.adjoint(tn::TensorNetwork{A}) where {A<:Quantum}
    tn = deepcopy(tn)

    reverse!(tn.interlayer)

    for tensor in tensors(tn)
        tensor .= conj(tensor)
    end

    return tn
end

## Numerical methods
function contract(a::TensorNetwork{<:Quantum}, b::TensorNetwork{<:Quantum}; kwargs...)
    contract(hcat(a, b); kwargs...)
end

# TODO look for more stable ways
function LinearAlgebra.norm(ψ::TensorNetwork{<:Quantum}, p::Real = 2; kwargs...)
    p != 2 && throw(ArgumentError("p=$p is not implemented yet"))

    return contract(hcat(ψ, ψ'); kwargs...) |> only |> sqrt |> abs
end

function LinearAlgebra.normalize!(
    ψ::TensorNetwork{<:Quantum},
    p::Real = 2;
    insert::Union{Nothing,Int} = nothing,
    kwargs...,
)
    norm = LinearAlgebra.norm(ψ; kwargs...)

    if isnothing(insert)
        # method 1: divide all tensors by (√v)^(1/n)
        n = length(ψ)
        norm ^= 1 / n
        for tensor in tensors(ψ)
            tensor ./= norm
        end
    else
        # method 2: divide only one tensor
        tensor = tensors(ψ, insert)
        tensor ./= norm
    end
end

fidelity(a, b; kwargs...) = abs(only(contract(a, b'; kwargs...)))^2

include("MP.jl")

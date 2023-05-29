using LinearAlgebra
using UUIDs: uuid4
using ValSplit
using Bijections

"""
    Quantum

Tensor Network [`Ansatz`](@ref) that has a notion of sites and directionality (input/output).
"""
abstract type Quantum <: Ansatz end

metadata(::Type{Quantum}) = NamedTuple{(:interlayer,),Tuple{Vector{Bijection{Int,Symbol}}}}

function checkmeta(::Type{Quantum}, tn::TensorNetwork)
    # TODO run this check depending if State or Operator
    length(tn.interlayer) >= 1 || return false

    # meta's indices exist
    all(bij -> values(bij) ⊆ labels(tn), tn.interlayer) || return false

    return true
end

abstract type Composite{Ts<:Tuple} <: Quantum end
Composite(@nospecialize(Ts::Type{<:Quantum}...)) = Composite{Tuple{Ts...}}
Base.fieldtypes(::Type{Composite{Ts}}) where {Ts} = fieldtypes(Ts)

metadata(A::Type{<:Composite}) = NamedTuple{(:layer, :interlayer),Tuple{NTuple{nlayers(A)},NTuple{nlayers(A) - 1}}}

function checkmeta(As::Type{<:Composite}, tn::TensorNetwork)
    for A in fieldtypes(As)
        tn_view = layers(tn, i)
        checkmeta(A, tn_view)
    end
end

nlayers(@nospecialize(T::Type{<:Composite})) = length(fieldtypes(T))

function layers(tn::TensorNetwork{<:Composite}, i)
    # TODO create view of TN
    meta = tn.layer[i]
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
plug(::T) where {T<:TensorNetwork} = plug(T)
plug(::Type{T}) where {T<:TensorNetwork} = plug(ansatz(T))

sites(tn::TensorNetwork) = collect(mapreduce(keys, ∪, tn.interlayer))

labels(tn::TensorNetwork, ::Val{:plug}) = unique(Iterators.flatmap(values, tn.interlayer))
labels(tn::TensorNetwork, ::Val{:plug}, site) = last(tn.interlayer)[site] # labels(tn, Val(:in), site) ∪ labels(tn, Val(:out), site)
labels(tn::TensorNetwork, ::Val{:virtual}) = setdiff(labels(tn, Val(:all)), labels(tn, Val(:plug)))

tensors(tn::TensorNetwork{<:Quantum}, site::Integer, args...) = tensors(plug(tn), tn, site, args...)
tensors(::Type{State}, tn::TensorNetwork{<:Quantum}, site) = select(tn, labels(tn, :plug, site)) |> only
@valsplit 4 tensors(T::Type{Operator}, tn::TensorNetwork{<:Quantum}, site, dir::Symbol) =
    throw(MethodError(sites, "dir=$dir not recognized"))

# TODO implement hcat when QA or QB <: Composite
function Base.hcat(A::TensorNetwork{QA}, B::TensorNetwork{QB}) where {QA<:Quantum,QB<:Quantum}
    issetequal(sites(A), sites(B)) ||
        throw(DimensionMismatch("A and B must contain the same set of sites in order to connect them"))

    # rename connector indices
    newinds = Dict([s => Symbol(uuid4()) for s in sites(A)])

    A = replace(A, [labels(A, :plug, site) => newinds[site] for site in sites(A)]...)
    B = replace(B, [labels(B, :plug, site) => newinds[site] for site in sites(B)]...)

    # rename inner indices of B to avoid hyperindices
    replace!(B, [i => Symbol(uuid4()) for i in labels(B, :inner)]...)

    # merge tensors and indices
    tn = TensorNetwork{Composite(QA, QB)}(
        [tensors(A)..., tensors(B)...];
        mergewith((a, b) -> a isa AbstractDict ? merge(a, b) : a, A.metadata, B.metadata)...,
    )

    return tn
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

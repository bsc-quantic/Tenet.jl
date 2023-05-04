using LinearAlgebra
using UUIDs: uuid4
using ValSplit

"""
    Quantum

Tensor Network [`Ansatz`](@ref) that has a notion of sites and directionality (input/output).
"""
abstract type Quantum <: Ansatz end

function checkmeta(::Type{Quantum}, tn::TensorNetwork)
    # meta exists
    haskey(tn.metadata, :plug) || return false

    # meta has correct type
    tn[:plug] isa AbstractDict{Tuple{Int,Symbol},Symbol} || return false
    all(∈(:in, :out) ∘ last, keys(tn[:plug])) || return false

    # meta's indices exist
    all(∈(keys(tn.indices)), values(tn[:plug])) || return false

    # meta's indices are not repeated
    allunique(values(tn[:plug])) || return false

    return true
end

sites(tn::TensorNetwork; dir::Symbol = :all) = sites(tn, dir)
@valsplit 2 sites(tn::TensorNetwork, dir::Symbol) = throw(MethodError(sites, "dir=$dir not recognized"))
sites(tn::TensorNetwork, ::Val{:all}) = unique(first.(keys(tn[:plug])))
sites(tn::TensorNetwork, ::Val{:in}) = first.(filter(==(:in) ∘ last, keys(tn[:plug])))
sites(tn::TensorNetwork, ::Val{:out}) = first.(filter(==(:out) ∘ last, keys(tn[:plug])))

labels(tn::TensorNetwork, ::Val{:plug}) = unique(values(tn[:plug]))
labels(tn::TensorNetwork, ::Val{:plug}, site) = labels(tn, Val(:in), site) ∪ labels(tn, Val(:out), site)
labels(tn::TensorNetwork, ::Val{:in}) = map(last, Iterators.filter((((_, dir), _),) -> dir === :in, tn[:plug]))
labels(tn::TensorNetwork, ::Val{:in}, site) = tn[:plug][(site, :in)]
labels(tn::TensorNetwork, ::Val{:out}) = map(last, Iterators.filter((((_, dir), _),) -> dir === :out, tn[:plug]))
labels(tn::TensorNetwork, ::Val{:out}, site) = tn[:plug][(site, :out)]
labels(tn::TensorNetwork, ::Val{:virtual}) = setdiff(labels(tn, Val(:all)), labels(tn, Val(:plug)))

tensors(tn::TensorNetwork{<:Quantum}, site::Integer) = select(tn, labels(tn, :plug, site))

abstract type Bounds end
abstract type Closed <: Bounds end
abstract type Open <: Bounds end

"""
    State{Bounds}

[`Quantum`](@ref) Tensor Network that only has outputs. Usually, it reflects the _state_ of a physical system.

Its adjoint only has inputs.
"""
abstract type State{B} <: Quantum where {B<:Bounds} end
bounds(::T) where {T<:State} = bounds(T)
bounds(::Type{<:State{B}}) where {B} = B

"""
    Operator{Bounds}

[`Quantum`](@ref) Tensor Network that has both inputs and outputs. Generally, it represents evolutionary processes of physical systems.
"""
abstract type Operator{B} <: Quantum where {B<:Bounds} end
bounds(::T) where {T<:Operator} = bounds(T)
bounds(::Type{<:Operator{B}}) where {B} = B

function Base.hcat(A::TensorNetwork{QA}, B::TensorNetwork{QB}) where {QA<:Quantum,QB<:Quantum}
    sites(A, :out) != sites(B, :in) &&
        throw(DimensionMismatch("sites(B,:in) must be equal to sites(A,:out) to connect them"))

    # rename connector indices
    newinds = Dict([s => Symbol(uuid4()) for s in sites(A, :out)])

    A = replace(A, [labels(A, :out, site) => newinds[site] for site in sites(A, :out)]...)
    B = replace(B, [labels(B, :in, site) => newinds[site] for site in sites(B, :in)]...)

    # remove plug metadata on connector indices
    for site in sites(A, :out)
        delete!(A[:plug], (site, :out))
    end
    for site in sites(B, :in)
        delete!(B[:plug], (site, :in))
    end

    # rename inner indices of B to avoid hyperindices
    replace!(B, [i => Symbol(uuid4()) for i in labels(B, :inner)]...)

    # merge tensors and indices
    tn = TensorNetwork{Tuple{QA,QB}}(; merge(A.metadata, B.metadata)...)
    append!(tn, A)
    append!(tn, B)

    return tn
end

Base.hcat(tns::TensorNetwork...) = reduce(hcat, tns)

function Base.adjoint(tn::TensorNetwork{A}) where {A<:Quantum}
    tn = deepcopy(tn)

    tmp = Dict((site, if dir === :in
        :out
    elseif dir === :out
        :in
    else
        dir
    end) => index for ((site, dir), index) in tn[:plug])
    merge!(tn[:plug], tmp)

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

include("MatrixProductState.jl")
include("MatrixProductOperator.jl")

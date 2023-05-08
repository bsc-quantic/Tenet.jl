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

tensors(tn::TensorNetwork{<:Quantum}, site::Integer, args...) = tensors(plug(tn), tn, site, args...)
tensors(::Type{State}, tn::TensorNetwork{<:Quantum}, site) = select(tn, labels(tn, :out, site)) |> only
@valsplit 4 tensors(T::Type{Operator}, tn::TensorNetwork{<:Quantum}, site, dir::Symbol) =
    throw(MethodError(sites, "dir=$dir not recognized"))
tensors(T::Type{Operator}, tn::TensorNetwork{<:Quantum}, site, ::Val{:in}) = select(tn, labels(tn, :in, site)) |> only
tensors(T::Type{Operator}, tn::TensorNetwork{<:Quantum}, site, ::Val{:out}) = select(tn, labels(tn, :out, site)) |> only

function Base.hcat(A::TensorNetwork{QA}, B::TensorNetwork{QB}) where {QA<:Quantum,QB<:Quantum}
    issetequal(sites(A, :out), sites(B, :in)) ||
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
    tn = TensorNetwork{Tuple{QA,QB}}([tensors(A)..., tensors(B)...]; mergewith(merge, A.metadata, B.metadata)...)

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

include("MP.jl")

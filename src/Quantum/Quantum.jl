using LinearAlgebra
using UUIDs: uuid4
using ValSplit
using Bijections: Bijection
using EinExprs: inds

"""
    Quantum <: Domain

Tensor Network `Ansatz` that has a notion of sites and directionality (input/output).
"""
struct Quantum <: Domain end

# NOTE info about the fields:
# - :layer => stores names of virtual indices belonging to each layer
# - :plug => stores names of physical indices belonging to each interlayer
# - :ansatz => stores ansatz trait of each layer
Base.fieldnames(::Type{Quantum}) = (:layer, :plug, :ansatz)
Base.fieldtypes(::Type{Quantum}) = (Vector{Set{Symbol}}, Vector{Bijection{Int,Symbol}}, Vector{Ansatz})

function checkdomain(tn::TensorNetwork{Quantum})
    # correct number of layers
    length(tn.layer) == length(tn.ansatz) ||
        throw(ArgumentError(":layer [$(length(tn.layer))] and :ansatz [$(length(tn.ansatz))] must have equal length"))

    # correct number of interlayers
    S = socket(tn)
    if S isa Property
        length(tn.plug) == length(tn.layer) - 1
    elseif S isa State
        length(tn.plug) == length(tn.layer)
    elseif S isa Operator
        length(tn.plug) == length(tn.layer) + 1
    else
        false
    end || throw(
        ArgumentError(
            "number of plugs [$(length(tn.plug))] does not match number of layers [$(length(tn.plug))] for socket $S",
        ),
    )

    # virtual indices exist
    all(is -> is ⊆ inds(tn), tn.layer) || throw(ArgumentError("all virtual indices in :layer must exist"))

    # physical indices exist
    all(is -> is ⊆ inds(tn), Iterators.map(values, tn.plug))

    # TODO virtual indices from different layers are not mixed

    # TODO call `checkansatz` on each layer
end

EinExprs.inds(tn::TensorNetwork{Quantum}, ::Val{:physical}) = unique(Iterators.flatten(Iterators.map(values, tn.plug)))
EinExprs.inds(tn::TensorNetwork{Quantum}, ::Val{:virtual}) = setdiff(inds(tn, Val(:all)), inds(tn, Val(:physical)))
EinExprs.inds(tn::TensorNetwork{Quantum}, ::Val{:plug}, site) = last(tn.plug)[site]
function EinExprs.inds(tn::TensorNetwork{Quantum}, ::Val{:out})
    socket(tn) isa State ||
        socket(tn) isa Operator ||
        throw(ErrorException("Must be a State or Operator to have output indices"))
    last(tn.plug) |> values
end
function EinExprs.inds(tn::TensorNetwork{Quantum}, ::Val{in})
    socket(tn) isa Operator || throw(ErrorException("Must be an Operator in order to have input indices"))
    first(tn.plug) |> values
end

function Base.replace!(tn::TensorNetwork{Quantum}, old_new::Pair{Symbol,Symbol})
    # replace indices in tensor network
    Base.@invoke replace!(tn::TensorNetwork, old_new::Pair{Symbol,Symbol})

    old, new = old_new

    # replace indices in layers
    for layer in tn.layer
        replace!(layer, old_new)
    end

    # replace indices in interlayers
    for plug in Iterators.filter(∋(old) ∘ values, tn.plug)
        site = plug(old)
        delete!(plug, site)
        plug[site] = new
    end

    return tn
end

"""
    nlayers(tn::TensorNetwork{Quantum})

Return the number of layers of a [`Quantum`](@ref) [`TensorNetwork`](@ref).
"""
nlayers(tn::TensorNetwork{Quantum}) = length(tn.layer)

"""
    sites(tn::TensorNetwork{Quantum})

Return the sites in which the [`TensorNetwork`](@ref) acts.
"""
sites(tn::TensorNetwork{Quantum}) = collect(mapreduce(keys, ∪, tn.plug))

"""
    adjoint(tn::TensorNetwork{Quantum})

Return the adjoint [`TensorNetwork`](@ref).

# Implementation details

The tensors are not transposed, just `conj` is applied to them.
"""
Base.adjoint(tn::TensorNetwork{Quantum}) = TensorNetwork{Quantum}(
    map(conj, tensors(tn));
    layer = reverse(tn.layer),
    plug = reverse(tn.plug),
    ansatz = reverse(tn.ansatz),
)

abstract type Boundary end
struct Open <: Boundary end
struct Periodic <: Boundary end
struct Infinite <: Boundary end

abstract type Socket end
struct Property <: Socket end
struct State <: Socket end
struct Operator <: Socket end

abstract type Ansatz{S<:Socket,B<:Boundary} end

"""
    boundary(::TensorNetwork)
    boundary(::Type{<:TensorNetwork})

Return the `Boundary` type of the [`TensorNetwork`](@ref). The following `Boundary`s are defined in `Tenet`:

  - `Open`
  - `Periodic`
  - `Infinite`
"""
function boundary end
boundary(::Type{<:Ansatz{S,B}}) where {S,B} = B()
boundary(tn::TensorNetwork{Quantum}) = boundary(first(tn.ansatz)) # TODO is this ok?

"""
    socket(::TensorNetwork{Quantum})
    socket(::Type{<:TensorNetwork})

Return the `Socket` type of the [`TensorNetwork`](@ref). The following `Socket`s are defined in `Tenet`:

  - `State` Only outputs.
  - `Operator` Inputs and outputs.
  - `Property` No inputs nor outputs.
"""
function socket end
socket(::Ansatz{S}) where {S} = S()
socket(::Type{<:Ansatz{S}}) where {S} = S()
socket(tn::TensorNetwork{Quantum}) = foldl(merge, Iterators.map(socket, tn.ansatz))

Base.merge(::State, ::State) = Property()
Base.merge(::State, ::Operator) = State()
Base.merge(::Operator, ::State) = State()
Base.merge(::Operator, ::Operator) = Operator()

# TODO finish `tensors` methods
"""
    tensors(tn::TensorNetwork{Quantum}, site::Integer)

Return the `Tensor` connected to the [`TensorNetwork`](@ref) on `site`.

See also: [`sites`](@ref).
"""
tensors(tn::TensorNetwork{Quantum}, site::Integer, args...) = tensors(socket(tn), tn, site, args...)
tensors(::Type{State}, tn::TensorNetwork{Quantum}, site) = select(tn, inds(tn, :plug, site)) |> only
@valsplit 4 tensors(T::Type{Operator}, tn::TensorNetwork{Quantum}, site, dir::Symbol) =
    throw(MethodError(sites, "dir=$dir not recognized"))

"""
    layer(tn::TensorNetwork{Quantum}, i)

Return a [`Quantum`](@ref) [`TensorNetwork`](@ref) that is a shallow copy of the ``i``-th layer of a .
"""
function layer(tn::TensorNetwork{Quantum}, i)
    1 <= i <= length(tn.layer) || throw(ArgumentError("Layer $i is out of bounds"))

    # select plugs
    S = socket(tn.ansatz[i])
    if S <: State && i ∈ [1, length(nlayers(tn))]
        throw(ErrorException("Layer #$i is a state but it is not a extreme layer"))
    end

    plug = if S <: State
        i == 1 ? [first(tn.plug)] : [last(tn.plug)]
    elseif S <: Operator
        # shift if first layer is a state
        socket(tn.ansatz[1]) <: State && (i = i - 1)
        tn.plug[i:i+1]
    end

    TensorNetwork{Quantum}(
        filter(tn.tensors) do tensor
            !isdisjoint(inds(tensor), tn.layer[i])
        end;
        layer = [tn.layer[i]],
        plug,
        ansatz = [tn.ansatz[i]],
    )
end

"""
    merge!(self::TensorNetwork{Quantum}, other::TensorNetwork{Quantum})

Fuse `other` into `self` by stacking layers.
"""
function Base.merge!(self::TensorNetwork{Quantum}, other::TensorNetwork{Quantum})
    # TODO check Socket combinations

    other = copy(other)

    # reindex `other` to avoid accidental hyperindices with `self`
    reindexvirtual!(other)

    # TODO reindex `other` to connect to `self` socket
    replace!(other, map(sites(other)) do site
        first(other.plug)[site] => last(self.plug)[site]
    end...)

    # merge attributes
    append!(self.layer, other.layer)
    append!(self.plug, other.plug)
    append!(self.ansatz, other.ansatz)

    @invoke merge!(self::TensorNetwork, other::TensorNetwork)
end

function reindexvirtual!(tn::TensorNetwork{Quantum})
    conflict = [i => Symbol(uuid4()) for i in inds(tn, :virtual)]
    replace!(tn, conflict...)
end

contract(a::TensorNetwork{Quantum}, b::TensorNetwork{Quantum}; kwargs...) = contract(merge(a, b); kwargs...)

"""
    norm(ψ::TensorNetwork{Quantum})

Compute the ``2``-norm of a [`Quantum`](@ref) [`TensorNetwork`](@ref).

See also: [`normalize!`](@ref).
"""
LinearAlgebra.norm(ψ::TensorNetwork{Quantum}, kwargs...) = contract(merge(ψ, ψ'); kwargs...) |> only |> sqrt |> abs

"""
    normalize!(ψ::TensorNetwork{Quantum}; insert::Union{Nothing,Int} = nothing)

In-place normalize a [`TensorNetwork`](@ref).

# Keyword Arguments

  - `insert` Choose the way the normalization is performed:

      + If `insert=nothing` (default), then all tensors are divided by ``\\sqrt[n]{\\lVert \\psi \\rVert_p}`` where `n` is the number of tensors.
      + If `insert isa Integer`, then the tensor connected to the site pointed by `insert` is divided by the norm.

    Both approaches are mathematically equivalent. Choose between them depending on the numerical properties.

See also: [`norm`](@ref).
"""
function LinearAlgebra.normalize!(ψ::TensorNetwork{Quantum}; insert::Union{Nothing,Int} = nothing, kwargs...)
    # TODO what if `nlayers` > 1?
    norm = LinearAlgebra.norm(ψ; kwargs...)

    if isnothing(insert)
        # method 1: divide all tensors by (√v)^(1/n)
        n = length(tensors(ψ))
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

"""
    fidelity(ψ,ϕ)

Compute the fidelity between states ``\\ket{\\psi}`` and ``\\ket{\\phi}``.
"""
fidelity(a, b; kwargs...) = abs(only(contract(a, b'; kwargs...)))^2

"""
    marginal(ψ, site)

Return the marginal quantum state of site.
"""
function marginal(ψ, site)
    tensor = tensors(ψ, site)
    index = inds(ψ, :plug, site)
    sum(tensor, inds = setdiff(inds(tensor), [index]))
end

Base.rand(A::Type{<:Ansatz}; kwargs...) = rand(Random.default_rng(), A; kwargs)

include("MP.jl")
include("PEP.jl")

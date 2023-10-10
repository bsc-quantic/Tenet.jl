using LinearAlgebra
using UUIDs: uuid4
using ValSplit
using Classes

"""
    QuantumTensorNetwork

Tensor Network that has a notion of sites and directionality (input/output).
"""
@class QuantumTensorNetwork <: TensorNetwork begin
    input::Vector{Symbol}
    output::Vector{Symbol}
end

inds(tn::absclass(QuantumTensorNetwork), ::Val{:in}) = tuple(tn.input...)
inds(tn::absclass(QuantumTensorNetwork), ::Val{:in}, site) = tn.input[site]
inds(tn::absclass(QuantumTensorNetwork), ::Val{:out}) = tuple(tn.output...)
inds(tn::absclass(QuantumTensorNetwork), ::Val{:out}, site) = tn.output[site]
inds(tn::absclass(QuantumTensorNetwork), ::Val{:physical}) = ∪(tn.input, tn.output)
inds(tn::absclass(QuantumTensorNetwork), ::Val{:virtual}) = setdiff(inds(tn, Val(:all)), inds(tn, Val(:physical)))

"""
    sites(tn::AbstractQuantumTensorNetwork, dir)

Return the sites in which the [`TensorNetwork`](@ref) acts.
"""
sites(tn::absclass(QuantumTensorNetwork)) = sites(tn, :in) ∪ sites(tn, :out)
function sites(tn::absclass(QuantumTensorNetwork), dir)
    if dir === :in
        firstindex(tn.input):lastindex(tn.input)
    elseif dir === :out
        firstindex(tn.output):lastindex(tn.output)
    else
        throw(MethodError("unknown dir=$dir"))
    end
end

function Base.replace!(tn::absclass(QuantumTensorNetwork), old_new::Pair{Symbol,Symbol})
    Base.@invoke replace!(tn::absclass(TensorNetwork), old_new::Pair{Symbol,Symbol})

    replace!(tn.input, old_new)
    replace!(tn.output, old_new)

    return tn
end

"""
    adjoint(tn::AbstractQuantumTensorNetwork)

Return the adjoint [`TensorNetwork`](@ref).

# Implementation details

The tensors are not transposed, just `conj!` is applied to them.
"""
function Base.adjoint(tn::absclass(QuantumTensorNetwork))
    tn = deepcopy(tn)

    # swap input/output
    temp = copy(tn.input)
    resize!(tn.input, length(tn.output))
    copy!(tn.input, tn.output)
    resize!(tn.output, length(temp))
    copy!(tn.output, temp)

    foreach(conj!, tensors(tn))

    return tn
end

function Base.merge!(self::absclass(QuantumTensorNetwork), other::absclass(QuantumTensorNetwork))
    sites(self, :out) == sites(other, :in) ||
        throw(DimensionMismatch("both `QuantumTensorNetwork`s must contain the same set of sites"))

    # copy to avoid mutation if reindex is needed
    # TODO deepcopy because `indices` are not correctly copied and it mutates
    other = deepcopy(other)

    # reindex other if needed
    if inds(self, set = :out) != inds(other, set = :in)
        replace!(other, map(splat(=>), zip(inds(other, set = :in), inds(self, set = :out))))
    end

    # reindex inner indices of `other` to avoid accidental hyperindices
    conflict = inds(self, set = :virtual) ∩ inds(other, set = :virtual)
    if !isempty(conflict)
        replace!(other, map(i -> i => Symbol(uuid4()), conflict))
    end

    @invoke merge!(self::absclass(TensorNetwork), other::absclass(TensorNetwork))

    # update i/o
    copy!(self.output, other.output)

    self
end

function contract(a::absclass(QuantumTensorNetwork), b::absclass(QuantumTensorNetwork); kwargs...)
    contract(merge(a, b); kwargs...)
end

# Plug trait
abstract type Plug end
struct Property <: Plug end
struct State <: Plug end
struct Dual <: Plug end
struct Operator <: Plug end

"""
    plug(::QuantumTensorNetwork)

Return the `Plug` type of the [`TensorNetwork`](@ref). The following `Plug`s are defined in `Tenet`:

  - `Property` No inputs nor outputs.
  - `State` Only outputs.
  - `Dual` Only inputs.
  - `Operator` Inputs and outputs.
"""
function plug(tn)
    if isempty(tn.input) && isempty(tn.output)
        Property()
    elseif isempty(tn.input)
        State()
    elseif isempty(tn.output)
        Dual()
    else
        Operator()
    end
end

# TODO look for more stable ways
"""
    norm(ψ::AbstractQuantumTensorNetwork, p::Real=2)

Compute the ``p``-norm of a [`QuantumTensorNetwork`](@ref).

See also: [`normalize!`](@ref).
"""
function LinearAlgebra.norm(ψ::absclass(QuantumTensorNetwork), p::Real = 2; kwargs...)
    p == 2 || throw(ArgumentError("p=$p is not implemented yet"))

    tn = merge(ψ, ψ')
    if plug(tn) isa Operator
        tn = tr(tn)
    end

    return contract(tn; kwargs...) |> only |> sqrt |> abs
end

"""
    normalize!(ψ::AbstractQuantumTensorNetwork, p::Real = 2; insert::Union{Nothing,Int} = nothing)

In-place normalize the [`TensorNetwork`](@ref).

# Keyword Arguments

  - `insert` Choose the way the normalization is performed:

      + If `insert=nothing` (default), then all tensors are divided by ``\\sqrt[n]{\\lVert \\psi \\rVert_p}`` where `n` is the number of tensors.
      + If `insert isa Integer`, then the tensor connected to the site pointed by `insert` is divided by the norm.

    Both approaches are mathematically equivalent. Choose between them depending on the numerical properties.

See also: [`norm`](@ref).
"""
function LinearAlgebra.normalize!(
    ψ::absclass(QuantumTensorNetwork),
    p::Real = 2;
    insert::Union{Nothing,Int} = nothing,
    kwargs...,
)
    norm = LinearAlgebra.norm(ψ, p; kwargs...)

    if isnothing(insert)
        # method 1: divide all tensors by (√v)^(1/n)
        n = length(tensors(ψ))
        norm ^= 1 / n
        for tensor in tensors(ψ)
            tensor ./= norm
        end
    else
        # method 2: divide only one tensor
        tensor = ψ.tensors[insert] # tensors(ψ, insert) # TODO fix this to match site?
        tensor ./= norm
    end
end

"""
    LinearAlgebra.tr(U::AbstractQuantumTensorNetwork)

Trace `U`: sum of diagonal elements if `U` is viewed as a matrix.

Depending on the result of `plug(U)`, different actions can be taken:

  - If `Property()`, the result of `contract(U)` will be a "scalar", for which the trace acts like the identity.
  - If `State()`, the result of `contract(U)` will be a "vector", for which the trace is undefined and will fail.
  - If `Operator()`, the input and output indices of `U` are connected.
"""
LinearAlgebra.tr(U::absclass(QuantumTensorNetwork)) = tr!(U)
tr!(U::absclass(QuantumTensorNetwork)) = tr!(plug(U), U)
tr!(::Property, scalar::absclass(QuantumTensorNetwork)) = scalar
function tr!(::Operator, U::absclass(QuantumTensorNetwork))
    sites(U, :in) == sites(U, :out) || throw(ArgumentError("input and output sites do not match"))
    copyto!(U.output, U.input)
    U
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
    plug(ψ) == State() || throw("unimplemented")

    siteindex = inds(ψ, :out, site)
    tensor = only(select(tn, siteindex))
    sum(tensor, inds = setdiff(inds(tensor), [siteindex]))
end

# Boundary trait
abstract type Boundary end
struct Open <: Boundary end
struct Periodic <: Boundary end

"""
    boundary(::QuantumTensorNetwork)

Return the `Boundary` type of the [`TensorNetwork`](@ref). The following `Boundary`s are defined in `Tenet`:

  - `Open`
  - `Periodic`
"""
function boundary end

abstract type Ansatz end

struct QTNSampler{A<:Ansatz} <: Random.Sampler{QuantumTensorNetwork}
    config::Dict{Symbol,Any}

    QTNSampler{A}(; kwargs...) where {A} = new{A}(kwargs)
end

Base.eltype(::QTNSampler{A}) where {A} = A

Base.getproperty(obj::QTNSampler, name::Symbol) = name === :config ? getfield(obj, :config) : obj.config[name]
Base.get(obj::QTNSampler, name, default) = get(obj.config, name, default)

Base.rand(A::Type{<:Ansatz}; kwargs...) = rand(Random.default_rng(), A; kwargs...)
Base.rand(rng::AbstractRNG, A::Type{<:Ansatz}; kwargs...) = rand(rng, QTNSampler{A}(; kwargs...))
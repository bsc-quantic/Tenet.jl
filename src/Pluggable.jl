
# TODO to new Quantum
# """
#     Quantum

# Tensor Network with a notion of "causality". This leads to the concept of sites and directionality (input/output).

# # Notes

#   - Indices are referenced by `Site`s.
# """
# struct Quantum <: AbstractQuantum
#     tn::TensorNetwork

#     # WARN keep them synchronized
#     sites::Dict{Site,Symbol}
#     # sitetensors::Dict{Site,Tensor}

#     function Quantum(tn::TensorNetwork, sites)
#         for (_, index) in sites
#             if !haskey(tn.indexmap, index)
#                 error("Index $index not found in TensorNetwork")
#             elseif index ∉ inds(tn; set=:open)
#                 error("Index $index must be open")
#             end
#         end

#         # sitetensors = map(sites) do (site, index)
#         #     site => tn[index]
#         # end |> Dict{Site,Tensor}

#         return new(tn, sites)
#     end
# end

# TODO to new Quantum
# Base.copy(tn::Quantum) = Quantum(copy(TensorNetwork(tn)), copy(tn.sites))
# Base.similar(tn::Quantum) = Quantum(similar(TensorNetwork(tn)), copy(tn.sites))
# Base.zero(tn::Quantum) = Quantum(zero(TensorNetwork(tn)), copy(tn.sites))

# TODO to new Quantum
# function Base.:(==)(a::AbstractQuantum, b::AbstractQuantum)
#     return Quantum(a).sites == Quantum(b).sites && ==(TensorNetwork(a), TensorNetwork(b))
# end

# TODO to new Quantum
# function Base.isapprox(a::AbstractQuantum, b::AbstractQuantum; kwargs...)
#     return Quantum(a).sites == Quantum(b).sites && isapprox(TensorNetwork(a), TensorNetwork(b); kwargs...)
# end

# TODO to new Quantum
# Base.summary(io::IO, tn::AbstractQuantum) = print(io, "$(ntensors(tn))-tensors Quantum")
# function Base.show(io::IO, tn::T) where {T<:AbstractQuantum}
#     return print(io, "$T (inputs=$(nsites(tn; set=:inputs)), outputs=$(nsites(tn; set=:outputs)))")
# end

# TODO to new Quantum
# tensors(kwargs::NamedTuple{(:at,)}, tn::AbstractQuantum) = only(tensors(tn; intersects=inds(tn; at=kwargs.at)))
# inds(kwargs::NamedTuple{(:at,)}, tn::AbstractQuantum) = Quantum(tn).sites[kwargs.at]

# TODO to new Quantum
function inds(kwargs::NamedTuple{(:set,)}, tn::AbstractQuantum)
    if kwargs.set === :physical
        return map(sites(tn)) do site
            inds(tn; at=site)::Symbol
        end
    elseif kwargs.set === :virtual
        return setdiff(inds(tn), inds(tn; set=:physical))
    elseif kwargs.set ∈ (:inputs, :outputs)
        return map(sites(tn; kwargs.set)) do site
            inds(tn; at=site)::Symbol
        end
    else
        return inds(TensorNetwork(tn); set=kwargs.set)
    end
end

# TODO to new Quantum
# `pop!` / `delete!` methods call this method
function Base.pop!(tn::AbstractQuantum, tensor::Tensor)
    pop!(TensorNetwork(tn), tensor)

    # TODO replace with `inds(tn; set=:physical)` when implemented
    targets = values(Quantum(tn).sites) ∩ inds(tensor)
    for target in targets
        rmsite!(tn, findfirst(==(target), Quantum(tn).sites))
    end

    return tensor
end

# TODO to new Quantum
function Base.replace!(tn::AbstractQuantum, old_new::Pair{Symbol,Symbol})
    tn = Quantum(tn)

    # replace indices in underlying Tensor Network
    replace!(TensorNetwork(tn), old_new)

    # replace indices in site information
    site = sites(tn; at=first(old_new))
    if !isnothing(site)
        rmsite!(tn, site)
        addsite!(tn, site, last(old_new))
    end

    return tn
end

# TODO to new Quantum
# FIXME return type should be the original type, not `Quantum`
function Base.replace!(tn::AbstractQuantum, old_new::Base.AbstractVecOrTuple{Pair{Symbol,Symbol}})
    tn = Quantum(tn)

    # replace indices in underlying Tensor Network
    replace!(TensorNetwork(tn), old_new)

    # replace indices in site information
    from, to = first.(old_new), last.(old_new)
    for (site, index) in tn.sites
        i = findfirst(==(index), from)
        if !isnothing(i)
            tn.sites[site] = to[i]
        end
    end

    return tn
end

# @deprecate inputs(tn::AbstractQuantum) sites(tn; set=:inputs)
# @deprecate outputs(tn::AbstractQuantum) sites(tn; set=:outputs)
# @deprecate ninputs(tn::AbstractQuantum) nsites(tn; set=:inputs)
# @deprecate noutputs(tn::AbstractQuantum) nsites(tn; set=:outputs)

# TODO to Ansatz
# """
#     lanes(q::AbstractQuantum)

# Return the lanes of a [`AbstractQuantum`](@ref) Tensor Network.
# """
# lanes(tn::AbstractQuantum) = unique!(Lane[Lane.(sites(tn; set=:inputs))..., Lane.(sites(tn; set=:outputs))...])

# TODO to new Quantum
# function addsite!(tn::AbstractQuantum, site, index)
#     tn = Quantum(tn)
#     if haskey(tn.sites, site)
#         error("Site $site already exists")
#     end

#     if index ∉ inds(tn; set=:open)
#         error("Index $index must be open")
#     end

#     return tn.sites[site] = index
# end

# TODO to new Quantum
# function rmsite!(tn::AbstractQuantum, site)
#     tn = Quantum(tn)
#     if !haskey(tn.sites, site)
#         error("Site $site does not exist")
#     end

#     return delete!(tn.sites, site)
# end

# TODO to new Quantum
# function sites(kwargs::NamedTuple{(:set,)}, tn::AbstractQuantum)
#     tn = Quantum(tn)
#     if kwargs.set === :all
#         sort!(collect(keys(tn.sites)))
#     elseif kwargs.set === :inputs
#         sort!(collect(Iterators.filter(isdual, keys(tn.sites))))
#     elseif kwargs.set === :outputs
#         sort!(collect(Iterators.filter(!isdual, keys(tn.sites))))
#     else
#         throw(ArgumentError("invalid set: $(kwargs.set)"))
#     end
# end

# TODO to new Quantum, and Circuit
# function sites(kwargs::@NamedTuple{at::Symbol}, tn::AbstractQuantum)
#     tn = Quantum(tn)
#     return findfirst(==(kwargs.at), tn.sites)
# end

# TODO to Stack
# """
#     Base.merge(a::AbstractQuantum, b::AbstractQuantum; reset=true)

# Merge multiple [`AbstractQuantum`](@ref) Tensor Networks. If `reset=true`, then all indices are renamed. If `reset=false`, then only the indices of the input/output sites are renamed.

# See also: [`merge!`](@ref), [`@reindex!`](@ref).
# """
# Base.merge(tns::AbstractQuantum...; kwargs...) = foldl((a, b) -> merge!(a, b; kwargs...), copy.(tns))
# Base.merge!(tns::AbstractQuantum...; kwargs...) = foldl((a, b) -> merge!(a, b; kwargs...), tns)

# TODO to Stack
# """
#     Base.merge!(::AbstractQuantum...; reset=true)

# Merge in-place multiple [`AbstractQuantum`](@ref) Tensor Networks. If `reset=true`, then all indices are renamed. If `reset=false`, then only the indices of the input/output sites are renamed.

# See also: [`merge`](@ref), [`@reindex!`](@ref).
# """
# function Base.merge!(a::AbstractQuantum, b::AbstractQuantum; reset=true)
#     @assert adjoint.(sites(b; set=:inputs)) ⊆ sites(a; set=:outputs) "Inputs of b must match outputs of a"
#     @assert isdisjoint(setdiff(sites(b; set=:outputs), adjoint.(sites(b; set=:inputs))), sites(a; set=:outputs)) "b cannot create new sites where is not connected"

#     @reindex! outputs(a) => inputs(b) reset = reset
#     merge!(TensorNetwork(a), TensorNetwork(b))

#     for site in sites(b; set=:inputs)
#         rmsite!(a, site')
#     end

#     for site in sites(b; set=:outputs)
#         addsite!(a, site, inds(b; at=site))
#     end

#     return a
# end

# TODO to new Quantum
# NOTE do not document because we might move it down to `Ansatz`
# LinearAlgebra.normalize(ψ::AbstractQuantum; kwargs...) = normalize!(copy(ψ); kwargs...)

# TODO to new Quantum
# """
#     LinearAlgebra.norm(::AbstractQuantum, p=2; kwargs...)

# Return the Lp-norm of a [`AbstractQuantum`](@ref) Tensor Network.

# !!! warning

#     Only L2-norm is implemented yet.
# """
# function LinearAlgebra.norm(ψ::AbstractQuantum, p::Real=2; kwargs...)
#     p == 2 || throw(ArgumentError("only L2-norm is implemented yet"))
#     return LinearAlgebra.norm2(ψ; kwargs...)
# end

# TODO to new Quantum
# LinearAlgebra.norm2(ψ::AbstractQuantum; kwargs...) = LinearAlgebra.norm2(socket(ψ), ψ; kwargs...)

# TODO to new Quantum
# function LinearAlgebra.norm2(::State, ψ::AbstractQuantum; kwargs...)
#     return abs(sqrt(only(contract(merge(ψ, ψ'); kwargs...))))
# end

# TODO to new Quantum
# function LinearAlgebra.norm2(::Operator, ψ::AbstractQuantum; kwargs...)
#     ψ, ϕ = Quantum(ψ), Quantum(ψ')

#     @reindex! outputs(ψ) => inputs(ϕ) reset = false
#     @reindex! inputs(ψ) => outputs(ϕ) reset = false
#     return abs(sqrt(only(contract(merge(TensorNetwork(ψ), TensorNetwork(ϕ)); kwargs...))))
# end

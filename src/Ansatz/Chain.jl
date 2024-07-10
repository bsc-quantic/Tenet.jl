using LinearAlgebra
using Random
using Muscle: gramschmidt!

struct Chain <: Ansatz
    super::Quantum
    boundary::Boundary
end

Base.copy(tn::Chain) = Chain(copy(Quantum(tn)), boundary(tn))

Base.similar(tn::Chain) = Chain(similar(Quantum(tn)), boundary(tn))
Base.zero(tn::Chain) = Chain(zero(Quantum(tn)), boundary(tn))

boundary(tn::Chain) = tn.boundary

MPS(arrays) = Chain(State(), Open(), arrays)
pMPS(arrays) = Chain(State(), Periodic(), arrays)
MPO(arrays) = Chain(Operator(), Open(), arrays)
pMPO(arrays) = Chain(Operator(), Periodic(), arrays)

alias(tn::Chain) = alias(socket(tn), boundary(tn), tn)
alias(::State, ::Open, ::Chain) = "MPS"
alias(::State, ::Periodic, ::Chain) = "pMPS"
alias(::Operator, ::Open, ::Chain) = "MPO"
alias(::Operator, ::Periodic, ::Chain) = "pMPO"

function Chain(tn::TensorNetwork, sites, args...; kwargs...)
    return Chain(Quantum(tn, sites), args...; kwargs...)
end

defaultorder(::Type{Chain}, ::State) = (:o, :l, :r)
defaultorder(::Type{Chain}, ::Operator) = (:o, :i, :l, :r)

function Chain(::State, boundary::Periodic, arrays::Vector{<:AbstractArray}; order=defaultorder(Chain, State()))
    @assert all(==(3) ∘ ndims, arrays) "All arrays must have 3 dimensions"
    issetequal(order, defaultorder(Chain, State())) ||
        throw(ArgumentError("order must be a permutation of $(String.(defaultorder(State())))"))

    n = length(arrays)
    symbols = [nextindex() for _ in 1:(2n)]

    _tensors = map(enumerate(arrays)) do (i, array)
        inds = map(order) do dir
            if dir == :o
                symbols[i]
            elseif dir == :r
                symbols[n + mod1(i, n)]
            elseif dir == :l
                symbols[n + mod1(i - 1, n)]
            else
                throw(ArgumentError("Invalid direction: $dir"))
            end
        end
        Tensor(array, inds)
    end

    sitemap = Dict(Site(i) => symbols[i] for i in 1:n)

    return Chain(Quantum(TensorNetwork(_tensors), sitemap), boundary)
end

function Chain(::State, boundary::Open, arrays::Vector{<:AbstractArray}; order=defaultorder(Chain, State()))
    @assert ndims(arrays[1]) == 2 "First array must have 2 dimensions"
    @assert all(==(3) ∘ ndims, arrays[2:(end - 1)]) "All arrays must have 3 dimensions"
    @assert ndims(arrays[end]) == 2 "Last array must have 2 dimensions"
    issetequal(order, defaultorder(Chain, State())) ||
        throw(ArgumentError("order must be a permutation of $(String.(defaultorder(Chain, State())))"))

    n = length(arrays)
    symbols = [nextindex() for _ in 1:(2n)]

    _tensors = map(enumerate(arrays)) do (i, array)
        _order = if i == 1
            filter(x -> x != :l, order)
        elseif i == n
            filter(x -> x != :r, order)
        else
            order
        end

        inds = map(_order) do dir
            if dir == :o
                symbols[i]
            elseif dir == :r
                symbols[n + mod1(i, n)]
            elseif dir == :l
                symbols[n + mod1(i - 1, n)]
            else
                throw(ArgumentError("Invalid direction: $dir"))
            end
        end
        Tensor(array, inds)
    end

    sitemap = Dict(Site(i) => symbols[i] for i in 1:n)

    return Chain(Quantum(TensorNetwork(_tensors), sitemap), boundary)
end

function Chain(::Operator, boundary::Periodic, arrays::Vector{<:AbstractArray}; order=defaultorder(Chain, Operator()))
    @assert all(==(4) ∘ ndims, arrays) "All arrays must have 4 dimensions"
    issetequal(order, defaultorder(Chain, Operator())) ||
        throw(ArgumentError("order must be a permutation of $(String.(defaultorder(Chain, Operator())))"))

    n = length(arrays)
    symbols = [nextindex() for _ in 1:(3n)]

    _tensors = map(enumerate(arrays)) do (i, array)
        inds = map(order) do dir
            if dir == :o
                symbols[i]
            elseif dir == :i
                symbols[i + n]
            elseif dir == :l
                symbols[2n + mod1(i - 1, n)]
            elseif dir == :r
                symbols[2n + mod1(i, n)]
            else
                throw(ArgumentError("Invalid direction: $dir"))
            end
        end
        Tensor(array, inds)
    end

    sitemap = Dict(Site(i) => symbols[i] for i in 1:n)
    merge!(sitemap, Dict(Site(i; dual=true) => symbols[i + n] for i in 1:n))

    return Chain(Quantum(TensorNetwork(_tensors), sitemap), boundary)
end

function Chain(::Operator, boundary::Open, arrays::Vector{<:AbstractArray}; order=defaultorder(Chain, Operator()))
    @assert ndims(arrays[1]) == 3 "First array must have 3 dimensions"
    @assert all(==(4) ∘ ndims, arrays[2:(end - 1)]) "All arrays must have 4 dimensions"
    @assert ndims(arrays[end]) == 3 "Last array must have 3 dimensions"
    issetequal(order, defaultorder(Chain, Operator())) ||
        throw(ArgumentError("order must be a permutation of $(String.(defaultorder(Chain, Operator())))"))

    n = length(arrays)
    symbols = [nextindex() for _ in 1:(3n - 1)]

    _tensors = map(enumerate(arrays)) do (i, array)
        _order = if i == 1
            filter(x -> x != :l, order)
        elseif i == n
            filter(x -> x != :r, order)
        else
            order
        end

        inds = map(_order) do dir
            if dir == :o
                symbols[i]
            elseif dir == :i
                symbols[i + n]
            elseif dir == :l
                symbols[2n + mod1(i - 1, n)]
            elseif dir == :r
                symbols[2n + mod1(i, n)]
            else
                throw(ArgumentError("Invalid direction: $dir"))
            end
        end
        Tensor(array, inds)
    end

    sitemap = Dict(Site(i) => symbols[i] for i in 1:n)
    merge!(sitemap, Dict(Site(i; dual=true) => symbols[i + n] for i in 1:n))

    return Chain(Quantum(TensorNetwork(_tensors), sitemap), boundary)
end

function Base.convert(::Type{Chain}, qtn::Product)
    arrs::Vector{Array} = arrays(TensorNetwork(qtn))
    arrs[1] = reshape(arrs[1], size(arrs[1])..., 1)
    arrs[end] = reshape(arrs[end], size(arrs[end])..., 1)
    map!(@view(arrs[2:(end - 1)]), @view(arrs[2:(end - 1)])) do arr
        reshape(arr, size(arr)..., 1, 1)
    end

    return Chain(socket(qtn), Open(), arrs)
end

leftsite(tn::Chain, site::Site) = leftsite(boundary(tn), tn, site)
function leftsite(::Open, tn::Chain, site::Site)
    return id(site) ∈ range(2, nlanes(tn)) ? Site(id(site) - 1; dual=isdual(site)) : nothing
end
leftsite(::Periodic, tn::Chain, site::Site) = Site(mod1(id(site) - 1, nlanes(tn)); dual=isdual(site))

rightsite(tn::Chain, site::Site) = rightsite(boundary(tn), tn, site)
function rightsite(::Open, tn::Chain, site::Site)
    return id(site) ∈ range(1, nlanes(tn) - 1) ? Site(id(site) + 1; dual=isdual(site)) : nothing
end
rightsite(::Periodic, tn::Chain, site::Site) = Site(mod1(id(site) + 1, nlanes(tn)); dual=isdual(site))

leftindex(tn::Chain, site::Site) = leftindex(boundary(tn), tn, site)
leftindex(::Open, tn::Chain, site::Site) = site == site"1" ? nothing : leftindex(Periodic(), tn, site)
leftindex(::Periodic, tn::Chain, site::Site) = inds(tn; bond=(site, leftsite(tn, site)))

rightindex(tn::Chain, site::Site) = rightindex(boundary(tn), tn, site)
function rightindex(::Open, tn::Chain, site::Site)
    return site == Site(nlanes(tn); dual=isdual(site)) ? nothing : rightindex(Periodic(), tn, site)
end
rightindex(::Periodic, tn::Chain, site::Site) = inds(tn; bond=(site, rightsite(tn, site)))

Base.adjoint(chain::Chain) = Chain(adjoint(Quantum(chain)), boundary(chain))

struct ChainSampler{B<:Boundary,S<:Socket,NT<:NamedTuple} <: Random.Sampler{Chain}
    parameters::NT

    ChainSampler{B,S}(; kwargs...) where {B,S} = new{B,S,typeof(values(kwargs))}(values(kwargs))
end

function Base.rand(A::Type{<:Chain}, B::Type{<:Boundary}, S::Type{<:Socket}; kwargs...)
    return rand(Random.default_rng(), A, B, S; kwargs...)
end

function Base.rand(rng::AbstractRNG, ::Type{A}, ::Type{B}, ::Type{S}; kwargs...) where {A<:Chain,B<:Boundary,S<:Socket}
    return rand(rng, ChainSampler{B,S}(; kwargs...), B, S)
end

function Base.rand(rng::Random.AbstractRNG, sampler::ChainSampler, ::Type{Open}, ::Type{State})
    n = sampler.parameters.n
    χ = sampler.parameters.χ
    p = get(sampler.parameters, :p, 2)
    T = get(sampler.parameters, :eltype, Float64)

    arrays::Vector{AbstractArray{T,N} where {N}} = map(1:n) do i
        χl, χr = let after_mid = i > n ÷ 2, i = (n + 1 - abs(2i - n - 1)) ÷ 2
            χl = min(χ, p^(i - 1))
            χr = min(χ, p^i)

            # swap bond dims after mid and handle midpoint for odd-length MPS
            (isodd(n) && i == n ÷ 2 + 1) ? (χl, χl) : (after_mid ? (χr, χl) : (χl, χr))
        end

        # fix for first site
        i == 1 && ((χl, χr) = (χr, 1))

        # orthogonalize by Gram-Schmidt algorithm
        A = gramschmidt!(rand(rng, T, χl, χr * p))

        A = reshape(A, χl, χr, p)
        permutedims(A, (3, 1, 2))
    end

    # reshape boundary sites
    arrays[1] = reshape(arrays[1], p, p)
    arrays[n] = reshape(arrays[n], p, p)

    # normalize state
    arrays[1] ./= sqrt(p)

    return Chain(State(), Open(), arrays)
end

# TODO let choose the orthogonality center
# TODO different input/output physical dims
function Base.rand(rng::Random.AbstractRNG, sampler::ChainSampler, ::Type{Open}, ::Type{Operator})
    n = sampler.parameters.n
    χ = sampler.parameters.χ
    p = get(sampler.parameters, :p, 2)
    T = get(sampler.parameters, :eltype, Float64)

    ip = op = p

    arrays::Vector{AbstractArray{T,N} where {N}} = map(1:n) do i
        χl, χr = let after_mid = i > n ÷ 2, i = (n + 1 - abs(2i - n - 1)) ÷ 2
            χl = min(χ, ip^(i - 1) * op^(i - 1))
            χr = min(χ, ip^i * op^i)

            # swap bond dims after mid and handle midpoint for odd-length MPS
            (isodd(n) && i == n ÷ 2 + 1) ? (χl, χl) : (after_mid ? (χr, χl) : (χl, χr))
        end

        shape = if i == 1
            (χr, ip, op)
        elseif i == n
            (χl, ip, op)
        else
            (χl, χr, ip, op)
        end

        # orthogonalize by Gram-Schmidt algorithm
        A = gramschmidt!(rand(rng, T, shape[1], prod(shape[2:end])))
        A = reshape(A, shape)

        (i == 1 || i == n) ? permutedims(A, (2, 3, 1)) : permutedims(A, (3, 4, 1, 2))
    end

    # normalize
    ζ = min(χ, ip * op)
    arrays[1] ./= sqrt(ζ)

    return Chain(Operator(), Open(), arrays)
end

Tenet.contract(tn::Chain, query::Symbol, args...; kwargs...) = contract!(copy(tn), Val(query), args...; kwargs...)
Tenet.contract!(tn::Chain, query::Symbol, args...; kwargs...) = contract!(tn, Val(query), args...; kwargs...)

"""
    Tenet.contract!(tn::Chain, ::Val{:between}, site1::Site, site2::Site; direction::Symbol = :left, delete_Λ = true)

For a given [`Chain`](@ref) tensor network, contracts the singular values Λ between two sites `site1` and `site2`.
The `direction` keyword argument specifies the direction of the contraction, and the `delete_Λ` keyword argument
specifies whether to delete the singular values tensor after the contraction.
"""
function Tenet.contract!(tn::Chain, ::Val{:between}, site1::Site, site2::Site; direction::Symbol=:left, delete_Λ=true)
    Λᵢ = tensors(tn; between=(site1, site2))
    Λᵢ === nothing && return tn

    if direction === :right
        Γᵢ₊₁ = tensors(tn; at=site2)
        replace!(TensorNetwork(tn), Γᵢ₊₁ => contract(Γᵢ₊₁, Λᵢ; dims=()))
    elseif direction === :left
        Γᵢ = tensors(tn; at=site1)
        replace!(TensorNetwork(tn), Γᵢ => contract(Λᵢ, Γᵢ; dims=()))
    else
        throw(ArgumentError("Unknown direction=:$direction"))
    end

    delete_Λ && delete!(TensorNetwork(tn), Λᵢ)

    return tn
end

canonize_site(tn::Chain, args...; kwargs...) = canonize_site!(deepcopy(tn), args...; kwargs...)
canonize_site!(tn::Chain, args...; kwargs...) = canonize_site!(boundary(tn), tn, args...; kwargs...)

# NOTE: in method == :svd the spectral weights are stored in a vector connected to the now virtual hyperindex!
function canonize_site!(::Open, tn::Chain, site::Site; direction::Symbol, method=:qr)
    left_inds = Symbol[]
    right_inds = Symbol[]

    virtualind = if direction === :left
        site == Site(1) && throw(ArgumentError("Cannot right-canonize left-most tensor"))
        push!(right_inds, leftindex(tn, site))

        site == Site(nsites(tn)) || push!(left_inds, rightindex(tn, site))
        push!(left_inds, Quantum(tn)[site])

        only(right_inds)
    elseif direction === :right
        site == Site(nsites(tn)) && throw(ArgumentError("Cannot left-canonize right-most tensor"))
        push!(right_inds, rightindex(tn, site))

        site == Site(1) || push!(left_inds, leftindex(tn, site))
        push!(left_inds, Quantum(tn)[site])

        only(right_inds)
    else
        throw(ArgumentError("Unknown direction=:$direction"))
    end

    tmpind = gensym(:tmp)
    if method === :svd
        svd!(TensorNetwork(tn); left_inds, right_inds, virtualind=tmpind)
    elseif method === :qr
        qr!(TensorNetwork(tn); left_inds, right_inds, virtualind=tmpind)
    else
        throw(ArgumentError("Unknown factorization method=:$method"))
    end

    contract!(TensorNetwork(tn), virtualind)
    replace!(TensorNetwork(tn), tmpind => virtualind)

    return tn
end

truncate(tn::Chain, args...; kwargs...) = truncate!(deepcopy(tn), args...; kwargs...)

"""
    truncate!(qtn::Chain, bond; threshold::Union{Nothing,Real} = nothing, maxdim::Union{Nothing,Int} = nothing)

Truncate the dimension of the virtual `bond`` of the [`Chain`](@ref) Tensor Network by keeping only the `maxdim` largest Schmidt coefficients or those larger than`threshold`.

# Notes

  - Either `threshold` or `maxdim` must be provided. If both are provided, `maxdim` is used.
  - The bond must contain the Schmidt coefficients, i.e. a site canonization must be performed before calling `truncate!`.
"""
function truncate!(qtn::Chain, bond; threshold::Union{Nothing,Real}=nothing, maxdim::Union{Nothing,Int}=nothing)
    # TODO replace for tensors(; between)
    vind = rightindex(qtn, bond[1])
    if vind != leftindex(qtn, bond[2])
        throw(ArgumentError("Invalid bond $bond"))
    end

    if vind ∉ inds(qtn; set=:hyper)
        throw(MissingSchmidtCoefficientsException(bond))
    end

    tensor = TensorNetwork(qtn)[vind]
    spectrum = parent(tensor)

    extent = collect(
        if !isnothing(maxdim)
            1:min(size(TensorNetwork(qtn), vind), maxdim)
        else
            1:size(TensorNetwork(qtn), vind)
        end,
    )

    # remove 0s from spectrum
    if isnothing(threshold)
        threshold = 1e-16
    end

    filter!(extent) do i
        abs(spectrum[i]) > threshold
    end

    slice!(TensorNetwork(qtn), vind, extent)

    return qtn
end

function isleftcanonical(qtn::Chain, site; atol::Real=1e-12)
    right_ind = rightindex(qtn, site)
    tensor = tensors(qtn; at=site)

    # we are at right-most site, we need to add an extra dummy dimension to the tensor
    if isnothing(right_ind)
        right_ind = gensym(:dummy)
        tensor = Tensor(reshape(parent(tensor), size(tensor)..., 1), (inds(tensor)..., right_ind))
    end

    # TODO is replace(conj(A)...) copying too much?
    contracted = contract(tensor, replace(conj(tensor), right_ind => gensym(:new_ind)))
    n = size(tensor, right_ind)
    identity_matrix = Matrix(I, n, n)

    return isapprox(contracted, identity_matrix; atol)
end

function isrightcanonical(qtn::Chain, site; atol::Real=1e-12)
    left_ind = leftindex(qtn, site)
    tensor = tensors(qtn; at=site)

    # we are at left-most site, we need to add an extra dummy dimension to the tensor
    if isnothing(left_ind)
        left_ind = gensym(:dummy)
        tensor = Tensor(reshape(parent(tensor), 1, size(tensor)...), (left_ind, inds(tensor)...))
    end

    #TODO is replace(conj(A)...) copying too much?
    contracted = contract(tensor, replace(conj(tensor), left_ind => gensym(:new_ind)))
    n = size(tensor, left_ind)
    identity_matrix = Matrix(I, n, n)

    return isapprox(contracted, identity_matrix; atol)
end

canonize(tn::Chain, args...; kwargs...) = canonize!(copy(tn), args...; kwargs...)
canonize!(tn::Chain, args...; kwargs...) = canonize!(boundary(tn), tn, args...; kwargs...)

"""
canonize(boundary::Boundary, tn::Chain)

Transform a `Chain` tensor network into the canonical form (Vidal form), that is,
we have the singular values matrix Λᵢ between each tensor Γᵢ₋₁ and Γᵢ.
"""
function canonize!(::Open, tn::Chain)
    Λ = Tensor[]

    # right-to-left QR sweep, get right-canonical tensors
    for i in nsites(tn):-1:2
        canonize_site!(tn, Site(i); direction=:left, method=:qr)
    end

    # left-to-right SVD sweep, get left-canonical tensors and singular values without reversing
    for i in 1:(nsites(tn) - 1)
        canonize_site!(tn, Site(i); direction=:right, method=:svd)

        # extract the singular values and contract them with the next tensor
        Λᵢ = pop!(TensorNetwork(tn), tensors(tn; between=(Site(i), Site(i + 1))))
        Aᵢ₊₁ = tensors(tn; at=Site(i + 1))
        replace!(TensorNetwork(tn), Aᵢ₊₁ => contract(Aᵢ₊₁, Λᵢ; dims=()))
        push!(Λ, Λᵢ)
    end

    for i in 2:nsites(tn) # tensors at i in "A" form, need to contract (Λᵢ)⁻¹ with A to get Γᵢ
        Λᵢ = Λ[i - 1] # singular values start between site 1 and 2
        A = tensors(tn; at=Site(i))
        Γᵢ = contract(A, Tensor(diag(pinv(Diagonal(parent(Λᵢ)); atol=1e-64)), inds(Λᵢ)); dims=())
        replace!(TensorNetwork(tn), A => Γᵢ)
        push!(TensorNetwork(tn), Λᵢ)
    end

    return tn
end

mixed_canonize(tn::Chain, args...; kwargs...) = mixed_canonize!(deepcopy(tn), args...; kwargs...)
mixed_canonize!(tn::Chain, args...; kwargs...) = mixed_canonize!(boundary(tn), tn, args...; kwargs...)

"""
    mixed_canonize!(boundary::Boundary, tn::Chain, center::Site)

Transform a `Chain` tensor network into the mixed-canonical form, that is,
for i < center the tensors are left-canonical and for i >= center the tensors are right-canonical,
and in the center there is a matrix with singular values.
"""
function mixed_canonize!(::Open, tn::Chain, center::Site) # TODO: center could be a range of sites
    # left-to-right QR sweep (left-canonical tensors)
    for i in 1:(id(center) - 1)
        canonize_site!(tn, Site(i); direction=:right, method=:qr)
    end

    # right-to-left QR sweep (right-canonical tensors)
    for i in nsites(tn):-1:(id(center) + 1)
        canonize_site!(tn, Site(i); direction=:left, method=:qr)
    end

    # center SVD sweep to get singular values
    canonize_site!(tn, center; direction=:left, method=:svd)

    return tn
end

"""
    LinearAlgebra.normalize!(tn::Chain, center::Site)

Normalizes the input [`Chain`](@ref) tensor network by transforming it
to mixed-canonized form with the given center site.
"""
function LinearAlgebra.normalize!(tn::Chain, root::Site; p::Real=2)
    mixed_canonize!(tn, root)
    normalize!(tensors(tn; between=(Site(id(root) - 1), root)), p)
    return tn
end

"""
    evolve!(qtn::Chain, gate)

Applies a local operator `gate` to the [`Chain`](@ref) tensor network.
"""
function evolve!(qtn::Chain, gate::Dense; threshold=nothing, maxdim=nothing, iscanonical=false, renormalize=false)
    # check gate is a valid operator
    if !(socket(gate) isa Operator)
        throw(ArgumentError("Gate must be an operator, but got $(socket(gate))"))
    end

    # TODO refactor out to `islane`?
    if !issetequal(adjoint.(inputs(gate)), outputs(gate))
        throw(ArgumentError("Gate inputs ($(inputs(gate))) and outputs ($(outputs(gate))) must be the same"))
    end

    # TODO refactor out to `canconnect`?
    if adjoint.(inputs(gate)) ⊈ outputs(qtn)
        throw(ArgumentError("Gate inputs ($(inputs(gate))) must be a subset of the TN sites ($(sites(qtn)))"))
    end

    if nlanes(gate) == 1
        evolve_1site!(qtn, gate)
    elseif nlanes(gate) == 2
        # check gate sites are contiguous
        # TODO refactor this out?
        gate_inputs = sort!(map(id, inputs(gate)))
        range = UnitRange(extrema(gate_inputs)...)

        range != gate_inputs && throw(ArgumentError("Gate lanes must be contiguous"))

        # TODO check correctly for periodic boundary conditions
        evolve_2site!(qtn, gate; threshold, maxdim, iscanonical, renormalize)
    else
        # TODO generalize for more than 2 lanes
        throw(ArgumentError("Invalid number of lanes $(nlanes(gate)), maximum is 2"))
    end

    return qtn
end

function evolve_1site!(qtn::Chain, gate::Dense)
    # shallow copy to avoid problems if errors in mid execution
    gate = copy(gate)

    contracting_index = gensym(:tmp)
    targetsite = only(inputs(gate))'

    # reindex contracting index
    replace!(TensorNetwork(qtn), inds(qtn; at=targetsite) => contracting_index)
    replace!(TensorNetwork(gate), inds(gate; at=targetsite') => contracting_index)

    # reindex output of gate to match TN sitemap
    replace!(TensorNetwork(gate), inds(gate; at=only(outputs(gate))) => inds(qtn; at=targetsite))

    # contract gate with TN
    merge!(TensorNetwork(qtn), TensorNetwork(gate))
    return contract!(TensorNetwork(qtn), contracting_index)
end

# TODO: Maybe rename iscanonical kwarg ?
function evolve_2site!(qtn::Chain, gate::Dense; threshold, maxdim, iscanonical=false, renormalize=false)
    # shallow copy to avoid problems if errors in mid execution
    gate = copy(gate)

    bond = sitel, siter = minmax(outputs(gate)...)
    left_inds::Vector{Symbol} = !isnothing(leftindex(qtn, sitel)) ? [leftindex(qtn, sitel)] : Symbol[]
    right_inds::Vector{Symbol} = !isnothing(rightindex(qtn, siter)) ? [rightindex(qtn, siter)] : Symbol[]

    virtualind::Symbol = inds(qtn; bond=bond)

    iscanonical ? contract_2sitewf!(qtn, bond) : contract!(TensorNetwork(qtn), virtualind)

    # reindex contracting index
    contracting_inds = [gensym(:tmp) for _ in inputs(gate)]
    replace!(
        TensorNetwork(qtn),
        map(zip(inputs(gate), contracting_inds)) do (site, contracting_index)
            inds(qtn; at=site') => contracting_index
        end,
    )
    replace!(
        TensorNetwork(gate),
        map(zip(inputs(gate), contracting_inds)) do (site, contracting_index)
            inds(gate; at=site) => contracting_index
        end,
    )

    # reindex output of gate to match TN sitemap
    for site in outputs(gate)
        if inds(qtn; at=site) != inds(gate; at=site)
            replace!(TensorNetwork(gate), inds(gate; at=site) => inds(qtn; at=site))
        end
    end

    # contract physical inds
    merge!(TensorNetwork(qtn), TensorNetwork(gate))
    contract!(TensorNetwork(qtn), contracting_inds)

    # decompose using SVD
    push!(left_inds, inds(qtn; at=sitel))
    push!(right_inds, inds(qtn; at=siter))

    if iscanonical
        unpack_2sitewf!(qtn, bond, left_inds, right_inds, virtualind)
    else
        svd!(TensorNetwork(qtn); left_inds, right_inds, virtualind)
    end
    # truncate virtual index
    if any(!isnothing, [threshold, maxdim])
        truncate!(qtn, bond; threshold, maxdim)

        # renormalize the bond
        if renormalize && iscanonical
            λ = tensors(qtn; between=bond)
            replace!(TensorNetwork(qtn), λ => normalize(λ))
        elseif renormalize && !iscanonical
            normalize!(qtn, bond[1])
        end
    end

    return qtn
end

"""
    contract_2sitewf!(ψ::Chain, bond)

For a given [`Chain`](@ref) in the canonical form, creates the two-site wave function θ with Λᵢ₋₁Γᵢ₋₁ΛᵢΓᵢΛᵢ₊₁,
where i is the `bond`, and replaces the Γᵢ₋₁ΛᵢΓᵢ tensors with θ.
"""
function contract_2sitewf!(ψ::Chain, bond)
    # TODO Check if ψ is in canonical form

    sitel, siter = bond # TODO Check if bond is valid
    (0 < id(sitel) < nsites(ψ) || 0 < id(siter) < nsites(ψ)) ||
        throw(ArgumentError("The sites in the bond must be between 1 and $(nsites(ψ))"))

    Λᵢ₋₁ = id(sitel) == 1 ? nothing : tensors(ψ; between=(Site(id(sitel) - 1), sitel))
    Λᵢ₊₁ = id(sitel) == nsites(ψ) - 1 ? nothing : tensors(ψ; between=(siter, Site(id(siter) + 1)))

    !isnothing(Λᵢ₋₁) && contract!(ψ, :between, Site(id(sitel) - 1), sitel; direction=:right, delete_Λ=false)
    !isnothing(Λᵢ₊₁) && contract!(ψ, :between, siter, Site(id(siter) + 1); direction=:left, delete_Λ=false)

    contract!(TensorNetwork(ψ), inds(ψ; bond=bond))

    return ψ
end

"""
    unpack_2sitewf!(ψ::Chain, bond)

For a given [`Chain`](@ref) that contains a two-site wave function θ in a bond, it decomposes θ into the canonical
form: Γᵢ₋₁ΛᵢΓᵢ, where i is the `bond`.
"""
function unpack_2sitewf!(ψ::Chain, bond, left_inds, right_inds, virtualind)
    # TODO Check if ψ is in canonical form

    sitel, siter = bond # TODO Check if bond is valid
    (0 < id(sitel) < nsites(ψ) || 0 < id(site_r) < nsites(ψ)) ||
        throw(ArgumentError("The sites in the bond must be between 1 and $(nsites(ψ))"))

    Λᵢ₋₁ = id(sitel) == 1 ? nothing : tensors(ψ; between=(Site(id(sitel) - 1), sitel))
    Λᵢ₊₁ = id(siter) == nsites(ψ) ? nothing : tensors(ψ; between=(siter, Site(id(siter) + 1)))

    # do svd of the θ tensor
    θ = tensors(ψ; at=sitel)
    U, s, Vt = svd(θ; left_inds, right_inds, virtualind)

    # contract with the inverse of Λᵢ and Λᵢ₊₂
    Γᵢ₋₁ =
        isnothing(Λᵢ₋₁) ? U : contract(U, Tensor(diag(pinv(Diagonal(parent(Λᵢ₋₁)); atol=1e-32)), inds(Λᵢ₋₁)); dims=())
    Γᵢ =
        isnothing(Λᵢ₊₁) ? Vt : contract(Tensor(diag(pinv(Diagonal(parent(Λᵢ₊₁)); atol=1e-32)), inds(Λᵢ₊₁)), Vt; dims=())

    delete!(TensorNetwork(ψ), θ)

    push!(TensorNetwork(ψ), Γᵢ₋₁)
    push!(TensorNetwork(ψ), s)
    push!(TensorNetwork(ψ), Γᵢ)

    return ψ
end

Tenet.__check_index_sizes(qtn::Chain) = Tenet.__check_index_sizes(TensorNetwork(qtn))

evolve(ψ::Chain, mpo::Chain) = evolve!(copy(ψ), mpo)

"""
    evolve!(ψ::Chain, mpo::Chain)

Applies a Matrix Product Operator (MPO) `mpo` to the [`Chain`](@ref).
"""
function evolve!(ψ::Chain, mpo::Chain)
    L = nsites(ψ)

    Tenet.@unsafe_region ψ begin
        for i in 1:L
            contractedind = inds(ψ; at=Site(i))
            t = contract(tensors(ψ; at=Site(i)), tensors(mpo; at=Site(i)); dims=(contractedind,))
            physicalind = inds(mpo; at=Site(i))

            # Fuse the two right legs of t into one
            if i == 1
                wanted_inds = (physicalind, rightindex(ψ, Site(i)), rightindex(mpo, Site(i)))
                new_inds = (contractedind, rightindex(ψ, Site(i)))
            elseif i < L
                wanted_inds = (
                    physicalind,
                    leftindex(ψ, Site(i)),
                    leftindex(mpo, Site(i)),
                    rightindex(ψ, Site(i)),
                    rightindex(mpo, Site(i)),
                )
                new_inds = (contractedind, leftindex(ψ, Site(i)), rightindex(ψ, Site(i)))
            else
                wanted_inds = (physicalind, leftindex(ψ, Site(i)), leftindex(mpo, Site(i)))
                new_inds = (contractedind, leftindex(ψ, Site(i)))
            end

            perm = Tenet.__find_index_permutation(wanted_inds, inds(t))
            t = permutedims(t, perm)

            t = Tensor(
                reshape(t, tuple(size(t, 1), [size(t, k) * size(t, k + 1) for k in 2:2:length(wanted_inds)]...)),
                new_inds,
            )

            replace!(TensorNetwork(ψ), tensors(ψ; at=Site(i)) => t)

            if i < L
                d = size(TensorNetwork(mpo), rightindex(mpo, Site(i)))
                Λᵢ = tensors(ψ; between=(Site(i), Site(i + 1)))
                Λᵢ = Tensor(diag(kron(Matrix(LinearAlgebra.I, d, d), diagm(parent(Λᵢ)))), inds(Λᵢ))
                replace!(TensorNetwork(ψ), tensors(ψ; between=(Site(i), Site(i + 1))) => Λᵢ)
            end
        end
    end

    return ψ
end

function expect(ψ::Chain, observables)
    # contract observable with TN
    ϕ = copy(ψ)
    for observable in observables
        evolve!(ϕ, observable)
    end

    # contract evolved TN with adjoint of original TN
    tn = merge!(TensorNetwork(ϕ), TensorNetwork(ψ'))

    return contract(tn)
end

overlap(a::Chain, b::Chain) = overlap(socket(a), a, socket(b), b)

# TODO fix optimal path
function overlap(::State, a::Chain, ::State, b::Chain)
    @assert issetequal(sites(a), sites(b)) "Ansatzes must have the same sites"

    b = copy(b)
    b = @reindex! outputs(a) => outputs(b)

    tn = merge(TensorNetwork(a), TensorNetwork(b'))
    return contract(tn)
end

# TODO optimize
overlap(a::Product, b::Chain) = contract(TensorNetwork(merge(Quantum(a), Quantum(b)')))
overlap(a::Chain, b::Product) = contract(TensorNetwork(merge(Quantum(a), Quantum(b)')))

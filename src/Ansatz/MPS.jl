using LinearAlgebra
using Random

"""
    MPS

A Matrix Product State (MPS) is a tensor network in a one-dimensional structure.
By default, we store the tensors in left-canonical form in the left of the `center` site and right-canonical form
in the right of the `center` site. The singular values are stored in the `λ` field.

# Keyword Arguments

  - `boundary::Boundary`: The boundary conditions of the MPS. It can be either `Open` or `Periodic`.
  - `λ::Vector{Tensor}`: The singular values of the MPS.
  - `center::Site`: The orthogonality center of the MPS.
"""
struct MPS <: Ansatz
    super::Quantum
    boundary::Boundary
    λ::Vector{Union{Nothing,Tensor}}
    center::Site
end

λ(tn::MPS) = tn.λ
lambdas(tn::MPS) = tn.λ

center(tn::MPS) = tn.center

Base.copy(tn::MPS) = MPS(copy(Quantum(tn)), boundary(tn))

Base.similar(tn::MPS) = MPS(similar(Quantum(tn)), boundary(tn))
Base.zero(tn::MPS) = MPS(zero(Quantum(tn)), boundary(tn))

boundary(tn::MPS) = tn.boundary

alias(tn::MPS) = alias(boundary(tn), tn)
alias(::Open, ::MPS) = "MPS"
alias(::Periodic, ::MPS) = "pMPS"

function MPS(tn::TensorNetwork, sites, args...; kwargs...)
    return MPS(Quantum(tn, sites), args...; kwargs...)
end

defaultorder(::Type{MPS}) = (:o, :l, :r)

function MPS(::State, boundary::Periodic, arrays::Vector{<:AbstractArray}, center::Site; order=defaultorder(MPS))
    @assert all(==(3) ∘ ndims, arrays) "All arrays must have 3 dimensions"
    issetequal(order, defaultorder(MPS)) ||
        throw(ArgumentError("order must be a permutation of $(String.(defaultorder(MPS)))"))
    center ∈ 1:n || throw(ArgumentError("Invalid center: $center"))

    n = length(arrays)
    gen = IndexCounter()
    symbols = [nextindex!(gen) for _ in 1:(2n)]

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

    mps = MPS(Quantum(TensorNetwork(_tensors), sitemap), boundary, [Tensor([1], [gensym(:λ)]) for _ in 1:n], center)

    # Check if the mps is in the proper canonical form
    𝟙ₗ = contract_env(mps, :left)
    isapprox(𝟙ₗ, Matrix(I, size(𝟙ₗ, 1), size(𝟙ₗ, 1))) ||
        throw(ArgumentError("The tensors of the MPS do not form an left-canonical isometry on for sites i < center"))

    𝟙ᵣ = contract_env(mps, :right)
    isapprox(𝟙ᵣ, Matrix(I, size(𝟙ᵣ, 1), size(𝟙ᵣ, 1))) ||
        throw(ArgumentError("The tensors of the MPS do not form an right-canonical isometry on for sites i >= center"))

    return mps
end

function MPS(boundary::Open, arrays::Vector{<:AbstractArray}, λ::Vector{Union{Nothing, Vector}}, center::Site; order=defaultorder(MPS))
    @assert ndims(arrays[1]) == 2 "First array must have 2 dimensions"
    @assert all(==(3) ∘ ndims, arrays[2:(end - 1)]) "All arrays must have 3 dimensions"
    @assert ndims(arrays[end]) == 2 "Last array must have 2 dimensions"
    issetequal(order, defaultorder(MPS)) ||
        throw(ArgumentError("order must be a permutation of $(String.(defaultorder(MPS)))"))

    n = length(arrays)
    gen = IndexCounter()
    symbols = [nextindex!(gen) for _ in 1:(2n)]

    (_tensors, λ) = (x -> (first.(x), last.(x)))(map(enumerate(arrays)) do (i, array)
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

        if i == length(arrays)
            return (Tensor(array, inds), nothing)
        else
            if λ[i] === nothing
                return (Tensor(array, inds), nothing)
            else
                return (Tensor(array, inds), Tensor(λ[i], [inds[findfirst(==(:r), _order)]]))
            end
        end
    end)

    sitemap = Dict(Site(i) => symbols[i] for i in 1:n)

    mps = MPS(Quantum(TensorNetwork(_tensors), sitemap), boundary, λ[1:end-1], center)

    for i in 1:n-1
        λ[i] !== nothing && push!(TensorNetwork(mps), λ[i])
    end

    # Check if the mps is in the proper canonical form
    𝟙ₗ = contract_env(mps, :left)
    @show Matrix{Float64}(I, size(𝟙ₗ, 1), size(𝟙ₗ, 1))
    isapprox(𝟙ₗ, Matrix{Float64}(I, size(𝟙ₗ, 1), size(𝟙ₗ, 1)); atol=1e-10) ||
        throw(ArgumentError("The tensors of the MPS do not form an left-canonical isometry on for sites i < center"))

    𝟙ᵣ = contract_env(mps, :right)
    isapprox(𝟙ᵣ, Matrix{Float64}(I, size(𝟙ᵣ, 1), size(𝟙ᵣ, 1)); atol=1e-10) ||
        throw(ArgumentError("The tensors of the MPS do not form an right-canonical isometry on for sites i >= center"))

    return mps
end

function Base.convert(::Type{MPS}, qtn::Product)
    arrs::Vector{Array} = arrays(qtn)
    arrs[1] = reshape(arrs[1], size(arrs[1])..., 1)
    arrs[end] = reshape(arrs[end], size(arrs[end])..., 1)
    map!(@view(arrs[2:(end - 1)]), @view(arrs[2:(end - 1)])) do arr
        reshape(arr, size(arr)..., 1, 1)
    end

    return MPS(socket(qtn), Open(), arrs)
end

leftsite(tn::MPS, site::Site) = leftsite(boundary(tn), tn, site)
function leftsite(::Open, tn::MPS, site::Site)
    return id(site) ∈ range(2, nlanes(tn)) ? Site(id(site) - 1; dual=isdual(site)) : nothing
end
leftsite(::Periodic, tn::MPS, site::Site) = Site(mod1(id(site) - 1, nlanes(tn)); dual=isdual(site))

rightsite(tn::MPS, site::Site) = rightsite(boundary(tn), tn, site)
function rightsite(::Open, tn::MPS, site::Site)
    return id(site) ∈ range(1, nlanes(tn) - 1) ? Site(id(site) + 1; dual=isdual(site)) : nothing
end
rightsite(::Periodic, tn::MPS, site::Site) = Site(mod1(id(site) + 1, nlanes(tn)); dual=isdual(site))

leftindex(tn::MPS, site::Site) = leftindex(boundary(tn), tn, site)
leftindex(::Open, tn::MPS, site::Site) = site == site"1" ? nothing : leftindex(Periodic(), tn, site)
leftindex(::Periodic, tn::MPS, site::Site) = inds(tn; bond=(site, leftsite(tn, site)))

rightindex(tn::MPS, site::Site) = rightindex(boundary(tn), tn, site)
function rightindex(::Open, tn::MPS, site::Site)
    return site == Site(nlanes(tn); dual=isdual(site)) ? nothing : rightindex(Periodic(), tn, site)
end
rightindex(::Periodic, tn::MPS, site::Site) = inds(tn; bond=(site, rightsite(tn, site)))

Base.adjoint(chain::MPS) = MPS(adjoint(Quantum(chain)), boundary(chain), reverse(lambdas(chain)), center(chain))

struct MPSSampler{B<:Boundary,NT<:NamedTuple} <: Random.Sampler{MPS}
    parameters::NT

    MPSSampler{B}(; kwargs...) where {B} = new{B,typeof(values(kwargs))}(values(kwargs))
end

function Base.rand(A::Type{<:MPS}, B::Type{<:Boundary}; kwargs...)
    return rand(Random.default_rng(), A, B; kwargs...)
end

function Base.rand(rng::AbstractRNG, ::Type{A}, ::Type{B}; kwargs...) where {A<:MPS,B<:Boundary}
    return rand(rng, MPSSampler{B}(; kwargs...), B)
end

function Base.rand(rng::Random.AbstractRNG, sampler::MPSSampler, ::Type{Open})
    n = sampler.parameters.n
    χ = sampler.parameters.χ
    p = get(sampler.parameters, :p, 2)
    T = get(sampler.parameters, :eltype, Float64)
    center = get(sampler.parameters, :center, Site(n ÷ 2))

    center = Site(id(center)-1) # TODO fix this

    arrays::Vector{AbstractArray{T,N} where {N}} = map(1:n) do i
        χl, χr = let after_mid = i > n ÷ 2, i = (n + 1 - abs(2i - n - 1)) ÷ 2
            χl = min(χ, p^(i - 1))
            χr = min(χ, p^i)

            # swap bond dims after mid and handle midpoint for odd-length MPS
            (isodd(n) && i == n ÷ 2 + 1) ? (χl, χl) : (after_mid ? (χr, χl) : (χl, χr))
        end

        # orthogonalize by QR factorization
        F = qr!(rand(rng, T, χl * p, χr))

        reshape(Matrix(F.Q), χl, p, χr)
    end

    λ = Vector{Union{Nothing, Vector}}(nothing, n - 1)

    # svd from right to left to get the right-canonical tensors
    for i in n:-1:id(center) + 2
        A = reshape(arrays[i], size(arrays[i], 1), size(arrays[i], 2), size(arrays[i], 3))
        A = permutedims(A, (3, 2, 1))
        A = reshape(A, size(A, 1) * size(A, 2), size(A, 3))
        U, V = Matrix(qr(A).Q), Matrix(qr(A).R)

        # if i > id(center) + 1
        arrays[i] = reshape(U, size(arrays[i], 3), size(arrays[i], 2), size(arrays[i], 1))
        arrays[i] = permutedims(arrays[i], (3, 2, 1))
        new = reshape(arrays[i-1], size(arrays[i-1], 1) * size(arrays[i-1], 2), size(arrays[i-1], 3)) * V
        arrays[i-1] = reshape(new, size(arrays[i-1], 1), size(arrays[i-1], 2), size(arrays[i-1], 3))
    end

    # svd in the center to get the singular values
    A = reshape(arrays[id(center)+1], size(arrays[id(center)+1], 1), size(arrays[id(center)+1], 2), size(arrays[id(center)+1], 3))
    A = permutedims(A, (3, 2, 1))
    A = reshape(A, size(A, 1) * size(A, 2), size(A, 3))
    U, s, V = svd(A)

    λ[id(center)] = s

    arrays[id(center)+1] = reshape(U, size(arrays[id(center)+1], 3), size(arrays[id(center)+1], 2), size(arrays[id(center)+1], 1))
    arrays[id(center)+1] = permutedims(arrays[id(center)+1], (3, 2, 1))
    new = reshape(arrays[id(center)], size(arrays[id(center)], 1) * size(arrays[id(center)], 2), size(arrays[id(center)], 3)) * V
    arrays[id(center)] = reshape(new, size(arrays[id(center)], 1), size(arrays[id(center)], 2), size(arrays[id(center)], 3))

    # reshape boundary sites
    arrays[1] = reshape(arrays[1], p, p)
    arrays[n] = reshape(arrays[n], p, p)

    mps = MPS(Open(), arrays, λ, center; order=(:l, :o, :r))
end

# """
#     Tenet.contract!(tn::MPS; between=(site1, site2), direction::Symbol = :left, delete_Λ = true)

# For a given [`MPS`](@ref) tensor network, contracts the singular values Λ between two sites `site1` and `site2`.
# The `direction` keyword argument specifies the direction of the contraction, and the `delete_Λ` keyword argument
# specifies whether to delete the singular values tensor after the contraction.
# """
@kwmethod contract(tn::MPS; between, direction, delete_Λ) = contract!(copy(tn); between, direction, delete_Λ)
@kwmethod function contract!(tn::MPS; between, direction, delete_Λ)
    site1, site2 = between
    Λᵢ = tensors(tn; between)
    Λᵢ === nothing && return tn

    if direction === :right
        Γᵢ₊₁ = tensors(tn; at=site2)
        replace!(tn, Γᵢ₊₁ => contract(Γᵢ₊₁, Λᵢ; dims=()))
    elseif direction === :left
        Γᵢ = tensors(tn; at=site1)
        replace!(tn, Γᵢ => contract(Λᵢ, Γᵢ; dims=()))
    else
        throw(ArgumentError("Unknown direction=:$direction"))
    end

    delete_Λ && delete!(TensorNetwork(tn), Λᵢ)

    return tn
end
@kwmethod contract(tn::MPS; between) = contract(tn; between, direction=:left, delete_Λ=true)
@kwmethod contract!(tn::MPS; between) = contract!(tn; between, direction=:left, delete_Λ=true)
@kwmethod contract(tn::MPS; between, direction) = contract(tn; between, direction, delete_Λ=true)
@kwmethod contract!(tn::MPS; between, direction) = contract!(tn; between, direction, delete_Λ=true)

"""
    contract_env(tn::MPS, direction::Symbol)

Contract the environment of an MPS and its adjoint from its `center` site to the `:left` or `:right`,
depending on the `direction` argument.
"""
function contract_env(mps::MPS, direction::Symbol)
    N = nsites(mps)
    adjoint_mps = mps'

    if direction === :left
        t1 = tensors(mps; at=Site(1))
        t2 = tensors(adjoint_mps; at=Site(1; dual=true))

        env = contract(t1, t2)

        for i in 2:(id(mps.center) - 1)
            # contract!(mps; between = (Site(i - 1), Site(i)), direction=:right)
            # contract!(adjoint_mps; between = (Site(i - 1; dual=true), Site(i; dual=true)), direction=:right)

            t1 = tensors(mps; at=Site(i))
            t2 = tensors(adjoint_mps; at=Site(i; dual=true))

            env = contract(env, t1, t2)
        end

        return env

    elseif direction === :right
        t1 = tensors(mps; at=Site(N))
        t2 = tensors(adjoint_mps; at=Site(N; dual=true))

        env = contract(t1, t2)

        for i in (N - 1):-1:(id(mps.center) + 1)
            # contract!(mps; between = (Site(i), Site(i + 1)), direction=:left)
            # contract!(adjoint_mps; between = (Site(i; dual=true), Site(i + 1; dual=true)), direction=:left)

            t1 = tensors(mps; at=Site(i))
            t2 = tensors(adjoint_mps; at=Site(i; dual=true))

            env = contract(env, t1, t2)
        end

        return env
    else
        throw(ArgumentError("Unknown direction=:$direction, must be either :left or :right"))
    end

    return env
end

"""
    compute_λ(tn::MPS)

Compute the singular values of the MPS tensor network.
"""
function compute_λ(tn::MPS)
    λ = Vector{Tensor}(undef, nsites(tn))

    for i in 1:nsites(tn)
        left = leftindex(tn, Site(i))
        right = rightindex(tn, Site(i))

        if left === nothing || right === nothing
            λ[i] = Tensor([1], [gensym(:λ)])
        else
            λ[i] = contract(tensors(tn; at=Site(i)), tensors(tn; at=Site(i + 1)); dims=(right, left))
        end
    end

    return λ
end

canonize_site(tn::MPS, args...; kwargs...) = canonize_site!(deepcopy(tn), args...; kwargs...)
canonize_site!(tn::MPS, args...; kwargs...) = canonize_site!(boundary(tn), tn, args...; kwargs...)

"""
    canonize_site(tn::MPS, site::Site; direction::Symbol, method=:qr)

Canonize the site `site` of the MPS tensor network `tn` in the `direction` with the `:qr` or `:svd` factorization method.
If `method` is `:svd`, the singular values are stored in a virtual hyperindex.
"""
function canonize_site!(::Open, tn::MPS, site::Site; direction::Symbol, method=:qr)
    left_inds = Symbol[]
    right_inds = Symbol[]

    virtualind = if direction === :left
        site == Site(1) && throw(ArgumentError("Cannot right-canonize left-most tensor"))
        push!(right_inds, leftindex(tn, site))

        site == Site(nsites(tn)) || push!(left_inds, rightindex(tn, site))
        push!(left_inds, inds(tn; at=site))

        only(right_inds)
    elseif direction === :right
        site == Site(nsites(tn)) && throw(ArgumentError("Cannot left-canonize right-most tensor"))
        push!(right_inds, rightindex(tn, site))

        site == Site(1) || push!(left_inds, leftindex(tn, site))
        push!(left_inds, inds(tn; at=site))

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

    contract!(tn, virtualind)
    replace!(tn, tmpind => virtualind)

    return tn
end

truncate(tn::MPS, args...; kwargs...) = truncate!(deepcopy(tn), args...; kwargs...)

"""
    truncate!(qtn::MPS, bond; threshold::Union{Nothing,Real} = nothing, maxdim::Union{Nothing,Int} = nothing)

Truncate the dimension of the virtual `bond`` of the [`MPS`](@ref) Tensor Network by keeping only the `maxdim` largest Schmidt coefficients or those larger than`threshold`.

# Notes

  - Either `threshold` or `maxdim` must be provided. If both are provided, `maxdim` is used.
  - The bond must contain the Schmidt coefficients, i.e. a site canonization must be performed before calling `truncate!`.
"""
function truncate!(qtn::MPS, bond; threshold::Union{Nothing,Real}=nothing, maxdim::Union{Nothing,Int}=nothing)
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
            1:min(size(qtn, vind), maxdim)
        else
            1:size(qtn, vind)
        end,
    )

    # remove 0s from spectrum
    if isnothing(threshold)
        threshold = 1e-16
    end

    filter!(extent) do i
        abs(spectrum[i]) > threshold
    end

    slice!(qtn, vind, extent)

    return qtn
end

function isleftcanonical(qtn::MPS, site; atol::Real=1e-12)
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

function isrightcanonical(qtn::MPS, site; atol::Real=1e-12)
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

canonize(tn::MPS, args...; kwargs...) = canonize!(copy(tn), args...; kwargs...)
canonize!(tn::MPS, args...; kwargs...) = canonize!(boundary(tn), tn, args...; kwargs...)

"""
    canonize(boundary::Boundary, tn::MPS)

Transform a `MPS` tensor network into the canonical form (Vidal form), that is,
we have the singular values matrix Λᵢ between each tensor Γᵢ₋₁ and Γᵢ.
"""
function canonize!(::Open, tn::MPS)
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
        replace!(tn, Aᵢ₊₁ => contract(Aᵢ₊₁, Λᵢ; dims=()))
        push!(Λ, Λᵢ)
    end

    for i in 2:nsites(tn) # tensors at i in "A" form, need to contract (Λᵢ)⁻¹ with A to get Γᵢ
        Λᵢ = Λ[i - 1] # singular values start between site 1 and 2
        A = tensors(tn; at=Site(i))
        Γᵢ = contract(A, Tensor(diag(pinv(Diagonal(parent(Λᵢ)); atol=1e-64)), inds(Λᵢ)); dims=())
        replace!(tn, A => Γᵢ)
        push!(TensorNetwork(tn), Λᵢ)
    end

    return tn
end

mixed_canonize(tn::MPS, args...; kwargs...) = mixed_canonize!(deepcopy(tn), args...; kwargs...)
mixed_canonize!(tn::MPS, args...; kwargs...) = mixed_canonize!(boundary(tn), tn, args...; kwargs...)

"""
    mixed_canonize!(boundary::Boundary, tn::MPS, center::Site)

Transform a `MPS` tensor network into the mixed-canonical form, that is,
for i < center the tensors are left-canonical and for i >= center the tensors are right-canonical,
and in the center there is a matrix with singular values.
"""
function mixed_canonize!(::Open, tn::MPS, center::Site) # TODO: center could be a range of sites
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
    LinearAlgebra.normalize!(tn::MPS, center::Site)

Normalizes the input [`MPS`](@ref) tensor network by transforming it
to mixed-canonized form with the given center site.
"""
function LinearAlgebra.normalize!(tn::MPS, root::Site; p::Real=2)
    mixed_canonize!(tn, root)
    normalize!(tensors(tn; between=(Site(id(root) - 1), root)), p)
    return tn
end

"""
    evolve!(qtn::MPS, gate)

Applies a local operator `gate` to the [`MPS`](@ref) tensor network.
"""
function evolve!(qtn::MPS, gate::Dense; threshold=nothing, maxdim=nothing, iscanonical=false, renormalize=false)
    # check gate is a valid operator
    if !(socket(gate) isa Operator)
        throw(ArgumentError("Gate must be an operator, but got $(socket(gate))"))
    end

    # TODO refactor out to `islane`?
    if !issetequal(adjoint.(sites(gate; set=:inputs)), sites(gate; set=:outputs))
        throw(
            ArgumentError(
                "Gate inputs ($(sites(gate; set=:inputs))) and outputs ($(sites(gate; set=:outputs))) must be the same"
            ),
        )
    end

    # TODO refactor out to `canconnect`?
    if adjoint.(sites(gate; set=:inputs)) ⊈ sites(qtn; set=:outputs)
        throw(
            ArgumentError("Gate inputs ($(sites(gate; set=:inputs))) must be a subset of the TN sites ($(sites(qtn)))")
        )
    end

    if nlanes(gate) == 1
        evolve_1site!(qtn, gate)
    elseif nlanes(gate) == 2
        # check gate sites are contiguous
        # TODO refactor this out?
        gate_inputs = sort!(id.(sites(gate; set=:inputs)))
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

function evolve_1site!(qtn::MPS, gate::Dense)
    # shallow copy to avoid problems if errors in mid execution
    gate = copy(gate)

    contracting_index = gensym(:tmp)
    targetsite = only(sites(gate; set=:inputs))'

    # reindex contracting index
    replace!(qtn, inds(qtn; at=targetsite) => contracting_index)
    replace!(gate, inds(gate; at=targetsite') => contracting_index)

    # reindex output of gate to match TN sitemap
    replace!(gate, inds(gate; at=only(sites(gate; set=:outputs))) => inds(qtn; at=targetsite))

    # contract gate with TN
    merge!(TensorNetwork(qtn), TensorNetwork(gate))
    return contract!(qtn, contracting_index)
end

# TODO: Maybe rename iscanonical kwarg ?
function evolve_2site!(qtn::MPS, gate::Dense; threshold, maxdim, iscanonical=false, renormalize=false)
    # shallow copy to avoid problems if errors in mid execution
    gate = copy(gate)

    bond = sitel, siter = minmax(sites(gate; set=:outputs)...)
    left_inds::Vector{Symbol} = !isnothing(leftindex(qtn, sitel)) ? [leftindex(qtn, sitel)] : Symbol[]
    right_inds::Vector{Symbol} = !isnothing(rightindex(qtn, siter)) ? [rightindex(qtn, siter)] : Symbol[]

    virtualind::Symbol = inds(qtn; bond=bond)

    iscanonical ? contract_2sitewf!(qtn, bond) : contract!(TensorNetwork(qtn), virtualind)

    # reindex contracting index
    contracting_inds = [gensym(:tmp) for _ in sites(gate; set=:inputs)]
    replace!(
        TensorNetwork(qtn),
        map(zip(sites(gate; set=:inputs), contracting_inds)) do (site, contracting_index)
            inds(qtn; at=site') => contracting_index
        end,
    )
    replace!(
        Quantum(gate),
        map(zip(sites(gate; set=:inputs), contracting_inds)) do (site, contracting_index)
            inds(gate; at=site) => contracting_index
        end,
    )

    # replace output indices of the gate for gensym indices
    output_inds = [gensym(:out) for _ in sites(gate; set=:outputs)]
    replace!(
        Quantum(gate),
        map(zip(sites(gate; set=:outputs), output_inds)) do (site, out)
            inds(gate; at=site) => out
        end,
    )

    # reindex output of gate to match TN sitemap
    for site in sites(gate; set=:outputs)
        if inds(qtn; at=site) != inds(gate; at=site)
            replace!(TensorNetwork(gate), inds(gate; at=site) => inds(qtn; at=site))
        end
    end

    # contract physical inds
    merge!(TensorNetwork(qtn), TensorNetwork(gate))
    contract!(qtn, contracting_inds)

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
            replace!(qtn, λ => normalize(λ)) # TODO this can be replaced by `normalize!(λ)`
        elseif renormalize && !iscanonical
            normalize!(qtn, bond[1])
        end
    end

    return qtn
end

"""
    contract_2sitewf!(ψ::MPS, bond)

For a given [`MPS`](@ref) in the canonical form, creates the two-site wave function θ with Λᵢ₋₁Γᵢ₋₁ΛᵢΓᵢΛᵢ₊₁,
where i is the `bond`, and replaces the Γᵢ₋₁ΛᵢΓᵢ tensors with θ.
"""
function contract_2sitewf!(ψ::MPS, bond)
    # TODO Check if ψ is in canonical form

    sitel, siter = bond # TODO Check if bond is valid
    (0 < id(sitel) < nsites(ψ) || 0 < id(siter) < nsites(ψ)) ||
        throw(ArgumentError("The sites in the bond must be between 1 and $(nsites(ψ))"))

    Λᵢ₋₁ = id(sitel) == 1 ? nothing : tensors(ψ; between=(Site(id(sitel) - 1), sitel))
    Λᵢ₊₁ = id(sitel) == nsites(ψ) - 1 ? nothing : tensors(ψ; between=(siter, Site(id(siter) + 1)))

    !isnothing(Λᵢ₋₁) && contract!(ψ; between=(Site(id(sitel) - 1), sitel), direction=:right, delete_Λ=false)
    !isnothing(Λᵢ₊₁) && contract!(ψ; between=(siter, Site(id(siter) + 1)), direction=:left, delete_Λ=false)

    contract!(ψ, inds(ψ; bond=bond))

    return ψ
end

"""
    unpack_2sitewf!(ψ::MPS, bond)

For a given [`MPS`](@ref) that contains a two-site wave function θ in a bond, it decomposes θ into the canonical
form: Γᵢ₋₁ΛᵢΓᵢ, where i is the `bond`.
"""
function unpack_2sitewf!(ψ::MPS, bond, left_inds, right_inds, virtualind)
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

function expect(ψ::MPS, observables)
    # contract observable with TN
    ϕ = copy(ψ)
    for observable in observables
        evolve!(ϕ, observable)
    end

    # contract evolved TN with adjoint of original TN
    tn = merge!(TensorNetwork(ϕ), TensorNetwork(ψ'))

    return contract(tn)
end

overlap(a::MPS, b::MPS) = overlap(socket(a), a, socket(b), b)

# TODO fix optimal path
function overlap(::State, a::MPS, ::State, b::MPS)
    @assert issetequal(sites(a), sites(b)) "Ansatzes must have the same sites"

    b = copy(b)
    b = @reindex! outputs(a) => outputs(b)

    tn = merge(TensorNetwork(a), TensorNetwork(b'))
    return contract(tn)
end

# TODO optimize
overlap(a::Product, b::MPS) = contract(merge(Quantum(a), Quantum(b)'))
overlap(a::MPS, b::Product) = contract(merge(Quantum(a), Quantum(b)'))

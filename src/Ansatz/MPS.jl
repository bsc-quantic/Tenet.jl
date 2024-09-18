using Random
using LinearAlgebra

abstract type AbstractMPS <: AbstractAnsatz end

mutable struct MPS <: AbstractMPS
    const tn::Ansatz
    form::Form
end

Ansatz(tn::MPS) = tn.tn

Base.copy(x::MPS) = MPS(copy(Ansatz(x)), form(x))
Base.similar(x::MPS) = MPS(similar(Ansatz(x)), form(x))
Base.zero(x::MPS) = MPS(zero(Ansatz(x)), form(x))

defaultorder(::Type{MPS}) = (:o, :l, :r)
boundary(::MPS) = Open()
form(tn::MPS) = tn.form

function MPS(arrays::Vector{<:AbstractArray}; order=defaultorder(MPS))
    @assert ndims(arrays[1]) == 2 "First array must have 2 dimensions"
    @assert all(==(3) ∘ ndims, arrays[2:(end - 1)]) "All arrays must have 3 dimensions"
    @assert ndims(arrays[end]) == 2 "Last array must have 2 dimensions"
    issetequal(order, defaultorder(MPS)) ||
        throw(ArgumentError("order must be a permutation of $(String.(defaultorder(MPS)))"))

    n = length(arrays)
    gen = IndexCounter()
    symbols = [nextindex!(gen) for _ in 1:(2n)]

    tn = TensorNetwork(
        map(enumerate(arrays)) do (i, array)
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
        end,
    )

    sitemap = Dict(Site(i) => symbols[i] for i in 1:n)
    qtn = Quantum(tn, sitemap)
    graph = path_graph(n)
    lattice = MetaGraph(graph, Site.(vertices(graph)) .=> nothing, map(x -> Site.(Tuple(x)) => nothing, edges(graph)))
    ansatz = Ansatz(qtn, lattice)
    return MPS(ansatz, NonCanonical())
end

function Base.convert(::Type{MPS}, tn::Product)
    @assert socket(tn) == State()

    arrs::Vector{Array} = arrays(tn)
    arrs[1] = reshape(arrs[1], size(arrs[1])..., 1)
    arrs[end] = reshape(arrs[end], size(arrs[end])..., 1)
    map!(@view(arrs[2:(end - 1)]), @view(arrs[2:(end - 1)])) do arr
        reshape(arr, size(arr)..., 1, 1)
    end

    return MPS(arrs)
end

Base.adjoint(tn::MPS) = MPS(adjoint(Ansatz(tn)), form(tn))

# TODO different input/output physical dims
# TODO let choose the orthogonality center
function Base.rand(rng::Random.AbstractRNG, ::Type{MPS}, n, χ; eltype=Float64, physical_dim=2)
    p = physical_dim
    T = eltype

    arrays::Vector{AbstractArray{T,N} where {N}} = map(1:n) do i
        χl, χr = let after_mid = i > n ÷ 2, i = (n + 1 - abs(2i - n - 1)) ÷ 2
            χl = min(χ, p^(i - 1))
            χr = min(χ, p^i)

            # swap bond dims after mid and handle midpoint for odd-length MPS
            (isodd(n) && i == n ÷ 2 + 1) ? (χl, χl) : (after_mid ? (χr, χl) : (χl, χr))
        end

        # orthogonalize by QR factorization
        F = lq!(rand(rng, T, χl, p * χr))

        reshape(Matrix(F.Q), χl, p, χr)
    end

    # reshape boundary sites
    arrays[1] = reshape(arrays[1], p, p)
    arrays[n] = reshape(arrays[n], p, p)

    return MPS(arrays; order=(:l, :o, :r))
end

# TODO deprecate contract(; between) and generalize it to AbstractAnsatz
"""
    Tenet.contract!(tn::MPS; between=(site1, site2), direction::Symbol = :left, delete_Λ = true)

For a given [`MPS`](@ref) tensor network, contracts the singular values Λ between two sites `site1` and `site2`.
The `direction` keyword argument specifies the direction of the contraction, and the `delete_Λ` keyword argument
specifies whether to delete the singular values tensor after the contraction.
"""
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

function sites(ψ::MPS, site::Site; dir)
    if dir === :left
        return site == site"1" ? nothing : Site(id(site) - 1)
    elseif dir === :right
        return site == Site(nsites(ψ)) ? nothing : Site(id(site) + 1)
    else
        throw(ArgumentError("Unknown direction for MPS = :$dir"))
    end
end

@kwmethod function inds(ψ::MPS; at, dir)
    if dir === :left && at == site"1"
        return nothing
    elseif dir === :right && at == Site(nlanes(ψ); dual=isdual(at))
        return nothing
    elseif dir ∈ (:left, :right)
        return inds(ψ; bond=(at, sites(ψ, at; dir)))
    else
        throw(ArgumentError("Unknown direction for MPS = :$dir"))
    end
end

function isleftcanonical(ψ::MPS, site; atol::Real=1e-12)
    right_ind = inds(ψ; at=site, dir=:right)
    tensor = tensors(ψ; at=site)

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

function isrightcanonical(ψ::MPS, site; atol::Real=1e-12)
    left_ind = inds(ψ; at=site, dir=:left)
    tensor = tensors(ψ; at=site)

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

# NOTE: in method == :svd the spectral weights are stored in a vector connected to the now virtual hyperindex!
function canonize_site!(ψ::MPS, site::Site; direction::Symbol, method=:qr)
    left_inds = Symbol[]
    right_inds = Symbol[]

    virtualind = if direction === :left
        site == Site(1) && throw(ArgumentError("Cannot right-canonize left-most tensor"))
        push!(right_inds, inds(ψ; at=site, dir=:left))

        site == Site(nsites(ψ)) || push!(left_inds, inds(ψ; at=site, dir=:right))
        push!(left_inds, inds(ψ; at=site))

        only(right_inds)
    elseif direction === :right
        site == Site(nsites(ψ)) && throw(ArgumentError("Cannot left-canonize right-most tensor"))
        push!(right_inds, inds(ψ; at=site, dir=:right))

        site == Site(1) || push!(left_inds, inds(ψ; at=site, dir=:left))
        push!(left_inds, inds(ψ; at=site))

        only(right_inds)
    else
        throw(ArgumentError("Unknown direction=:$direction"))
    end

    tmpind = gensym(:tmp)
    if method === :svd
        svd!(ψ; left_inds, right_inds, virtualind=tmpind)
    elseif method === :qr
        qr!(ψ; left_inds, right_inds, virtualind=tmpind)
    else
        throw(ArgumentError("Unknown factorization method=:$method"))
    end

    contract!(ψ, virtualind)
    replace!(ψ, tmpind => virtualind)

    return ψ
end

"""
    canonize!(tn::MPS)

Transform a [`MPS`](@ref) tensor network into the canonical form (Vidal form); i.e. the singular values matrix Λᵢ between each tensor Γᵢ₋₁ and Γᵢ.
"""
function canonize!(ψ::MPS)
    Λ = Tensor[]

    # right-to-left QR sweep, get right-canonical tensors
    for i in nsites(ψ):-1:2
        canonize_site!(ψ, Site(i); direction=:left, method=:qr)
    end

    # left-to-right SVD sweep, get left-canonical tensors and singular values without reversing
    for i in 1:(nsites(ψ) - 1)
        canonize_site!(ψ, Site(i); direction=:right, method=:svd)

        # extract the singular values and contract them with the next tensor
        Λᵢ = pop!(ψ, tensors(ψ; between=(Site(i), Site(i + 1))))
        Aᵢ₊₁ = tensors(ψ; at=Site(i + 1))
        replace!(ψ, Aᵢ₊₁ => contract(Aᵢ₊₁, Λᵢ; dims=()))
        push!(Λ, Λᵢ)
    end

    for i in 2:nsites(ψ) # tensors at i in "A" form, need to contract (Λᵢ)⁻¹ with A to get Γᵢ
        Λᵢ = Λ[i - 1] # singular values start between site 1 and 2
        A = tensors(ψ; at=Site(i))
        Γᵢ = contract(A, Tensor(diag(pinv(Diagonal(parent(Λᵢ)); atol=1e-64)), inds(Λᵢ)); dims=())
        replace!(ψ, A => Γᵢ)
        push!(ψ, Λᵢ)
    end

    return ψ
end

mixed_canonize(tn::MPS, args...; kwargs...) = mixed_canonize!(deepcopy(tn), args...; kwargs...)

# TODO mixed_canonize! at bond
"""
    mixed_canonize!(tn::MPS, orthog_center)

Transform a [`MPS`](@ref) tensor network into the mixed-canonical form, that is,
for `i < orthog_center` the tensors are left-canonical and for `i >= orthog_center` the tensors are right-canonical,
and in the `orthog_center` there is a matrix with singular values.
"""
function mixed_canonize!(tn::MPS, orthog_center)
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

# TODO normalize! methods
function LinearAlgebra.normalize!(ψ::MPS, orthog_center=site"1")
    mixed_canonize!(ψ, orthog_center)
    normalize!(tensors(tn; between=(Site(id(root) - 1), root)), 2)
    return ψ
end

using Random
using LinearAlgebra
using Graphs
using BijectiveDicts: BijectiveIdDict

abstract type AbstractMPO <: AbstractAnsatz end
abstract type AbstractMPS <: AbstractMPO end

"""
    MPS <: AbstractAnsatz

A Matrix Product State [`Ansatz`](@ref) Tensor Network.
"""
mutable struct MPS <: AbstractMPS
    const tn::Ansatz
    form::Form
end

"""
    MPO <: AbstractAnsatz

A Matrix Product Operator (MPO) [`Ansatz`](@ref) Tensor Network.
"""
mutable struct MPO <: AbstractMPO
    const tn::Ansatz
    form::Form
end

Ansatz(tn::Union{MPS,MPO}) = tn.tn

boundary(::Union{MPS,MPO}) = Open()
form(tn::Union{MPS,MPO}) = tn.form

Base.copy(x::T) where {T<:Union{MPS,MPO}} = T(copy(Ansatz(x)), form(x))
Base.similar(x::T) where {T<:Union{MPS,MPO}} = T(similar(Ansatz(x)), form(x))
Base.zero(x::T) where {T<:Union{MPS,MPO}} = T(zero(Ansatz(x)), form(x))

defaultorder(::Type{<:AbstractMPS}) = (:o, :l, :r)
defaultorder(::Type{<:AbstractMPO}) = (:o, :i, :l, :r)

MPS(arrays; form::Form=NonCanonical(), kwargs...) = MPS(form, arrays; kwargs...)
function MPS(arrays::Vector{<:AbstractArray}, λ; form::Form=Canonical(), kwargs...)
    return MPS(form, arrays, λ; kwargs...)
end

"""
    MPS(arrays::Vector{<:AbstractArray}; order=defaultorder(MPS))

Create a [`NonCanonical`](@ref) [`MPS`](@ref) from a vector of arrays.

# Keyword Arguments

  - `order` The order of the indices in the arrays. Defaults to `(:o, :l, :r)`.
"""
function MPS(::NonCanonical, arrays; order=defaultorder(MPS), check=true)
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
    mapping = BijectiveIdDict{Site,Int}(Pair{Site,Int}[site => i for (i, site) in enumerate(lanes(qtn))])
    lattice = Lattice(mapping, graph)
    ansatz = Ansatz(qtn, lattice)
    return MPS(ansatz, NonCanonical())
end

function MPS(form::MixedCanonical, arrays; order=defaultorder(MPS), check=true)
    mps = MPS(arrays; form=NonCanonical(), order, check)
    mps.form = form

    # Check mixed canonical form
    check && check_form(mps)

    return mps
end

function MPS(::Canonical, arrays, λ; order=defaultorder(MPS), check=true)
    @assert length(λ) == length(arrays) - 1 "Number of λ tensors must be one less than the number of arrays"
    @assert all(==(1) ∘ ndims, λ) "All λ tensors must be Vectors"

    mps = MPS(arrays; form=NonCanonical(), order, check)
    mps.form = Canonical()

    # Create tensors from 'λ'
    map(enumerate(λ)) do (i, array)
        tensor = Tensor(array, (inds(mps; at=Site(i), dir=:right),))
        push!(mps, tensor)
    end

    # Check canonical form by contracting Γ and λ tensors and checking their orthogonality
    check && check_form(mps)

    return mps
end

check_form(mps::AbstractMPO) = check_form(form(mps), mps)

function check_form(config::MixedCanonical, mps::AbstractMPO)
    orthog_center = config.orthog_center
    for i in 1:nsites(mps)
        if i < id(orthog_center) # Check left-canonical tensors
            isisometry(mps, Site(i); dir=:right) || throw(ArgumentError("Tensors are not left-canonical"))
        elseif i > id(orthog_center) # Check right-canonical tensors
            isisometry(mps, Site(i); dir=:left) || throw(ArgumentError("Tensors are not right-canonical"))
        end
    end

    return true
end

function check_form(::Canonical, mps::AbstractMPO)
    for i in 1:nsites(mps)
        if i > 1
            !isisometry(contract(mps; between=(Site(i - 1), Site(i)), direction=:right), Site(i); dir=:right)
            throw(ArgumentError("Can not form a left-canonical tensor in Site($i) from Γ and λ contraction."))
        end

        if i < nsites(mps) &&
            !isisometry(contract(mps; between=(Site(i), Site(i + 1)), direction=:left), Site(i); dir=:left)
            throw(ArgumentError("Can not form a right-canonical tensor in Site($i) from Γ and λ contraction."))
        end
    end

    return true
end

"""
    MPO(arrays::Vector{<:AbstractArray}; order=defaultorder(MPO))

Create a [`NonCanonical`](@ref) [`MPO`](@ref) from a vector of arrays.

# Keyword Arguments

  - `order` The order of the indices in the arrays. Defaults to `(:o, :i, :l, :r)`.
"""
function MPO(arrays::Vector{<:AbstractArray}; order=defaultorder(MPO))
    @assert ndims(arrays[1]) == 3 "First array must have 3 dimensions"
    @assert all(==(4) ∘ ndims, arrays[2:(end - 1)]) "All arrays must have 4 dimensions"
    @assert ndims(arrays[end]) == 3 "Last array must have 3 dimensions"
    issetequal(order, defaultorder(MPO)) ||
        throw(ArgumentError("order must be a permutation of $(String.(defaultorder(MPO)))"))

    n = length(arrays)
    gen = IndexCounter()
    symbols = [nextindex!(gen) for _ in 1:(3n - 1)]

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
        end,
    )

    sitemap = Dict(Site(i) => symbols[i] for i in 1:n)
    merge!(sitemap, Dict(Site(i; dual=true) => symbols[i + n] for i in 1:n))
    qtn = Quantum(tn, sitemap)
    graph = path_graph(n)
    mapping = BijectiveIdDict{Site,Int}(Pair{Site,Int}[site => i for (i, site) in enumerate(lanes(qtn))])
    lattice = Lattice(mapping, graph)
    ansatz = Ansatz(qtn, lattice)
    return MPO(ansatz, NonCanonical())
end

"""
    MPS(::typeof(identity), n::Integer; physdim=2, maxdim=physdim^(n ÷ 2))

Returns an [`MPS`](@ref) of `n` sites whose tensors are initialized to COPY-tensors.

# Keyword Arguments

  - `physdim` The physical or output dimension of each site. Defaults to 2.
  - `maxdim` The maximum bond dimension. Defaults to `physdim^(n ÷ 2)`.
"""
function MPS(::typeof(identity), n::Integer; physdim=2, maxdim=physdim^(n ÷ 2))
    # Create bond dimensions until the middle of the MPS considering maxdim
    virtualdims = min.(maxdim, physdim .^ (1:(n ÷ 2)))

    # Complete the bond dimensions of the other half of the MPS
    virtualdims = vcat(virtualdims, virtualdims[(isodd(n) ? end : end - 1):-1:1])

    # Create each site dimensions in default order (:o, :l, :r)
    arraysdims = [[physdim, virtualdims[1]]]
    append!(arraysdims, [[physdim, virtualdims[i], virtualdims[i + 1]] for i in 1:(length(virtualdims) - 1)])
    push!(arraysdims, [physdim, virtualdims[end]])

    # Create the MPS with copy-tensors according to the tensors dimensions
    return MPS(
        map(arraysdims) do arrdims
            arr = zeros(ComplexF64, arrdims...)
            deltas = [fill(i, length(arrdims)) for i in 1:physdim]
            broadcast(delta -> arr[delta...] = 1.0, deltas)
            arr
        end,
    )
end

function Base.convert(::Type{T}, tn::Product) where {T<:AbstractMPO}
    @assert socket(tn) == State()

    arrs::Vector{Array} = arrays(tn)
    arrs[1] = reshape(arrs[1], size(arrs[1])..., 1)
    arrs[end] = reshape(arrs[end], size(arrs[end])..., 1)
    map!(@view(arrs[2:(end - 1)]), @view(arrs[2:(end - 1)])) do arr
        reshape(arr, size(arr)..., 1, 1)
    end

    return T(arrs)
end

# TODO can this be better written? or even generalized to AbstractAnsatz?
Base.adjoint(tn::T) where {T<:AbstractMPO} = T(adjoint(Ansatz(tn)), form(tn))

# TODO different input/output physical dims
# TODO let choose the orthogonality center
# TODO add form information
"""
    Base.rand(rng::Random.AbstractRNG, ::Type{MPS}; n, maxdim, eltype=Float64, physdim=2)

Create a random [`MPS`](@ref) Tensor Network in the MixedCanonical form where all tensors are right-canonical (ortogonality
center at the first site). In order to avoid norm explosion issues, the tensors are orthogonalized by LQ factorization.

# Keyword Arguments

  - `n` The number of sites.
  - `maxdim` The maximum bond dimension.
  - `eltype` The element type of the tensors. Defaults to `Float64`.
  - `physdim` The physical or output dimension of each site. Defaults to 2.
"""
function Base.rand(rng::Random.AbstractRNG, ::Type{MPS}; n, maxdim, eltype=Float64, physdim=2)
    p = physdim
    T = eltype
    χ = maxdim

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

    return MPS(arrays; order=(:l, :o, :r), form=MixedCanonical(Site(1)))
end

# TODO different input/output physical dims
# TODO let choose the orthogonality center
"""
    Base.rand(rng::Random.AbstractRNG, ::Type{MPO}; n, maxdim, eltype=Float64, physdim=2)

Create a random [`MPO`](@ref) Tensor Network.
In order to avoid norm explosion issues, the tensors are orthogonalized by QR factorization so its normalized and mixed canonized to the last site.

# Keyword Arguments

  - `n` The number of sites.
  - `maxdim` The maximum bond dimension.
  - `eltype` The element type of the tensors. Defaults to `Float64`.
  - `physdim` The physical or output dimension of each site. Defaults to 2.
"""
function Base.rand(rng::Random.AbstractRNG, ::Type{MPO}; n, maxdim, eltype=Float64, physdim=2)
    T = eltype
    ip = op = physdim
    χ = maxdim

    arrays::Vector{AbstractArray{T,N} where {N}} = map(1:n) do i
        χl, χr = let after_mid = i > n ÷ 2, i = (n + 1 - abs(2i - n - 1)) ÷ 2
            χl = min(χ, ip^(i - 1) * op^(i - 1))
            χr = min(χ, ip^i * op^i)

            # swap bond dims after mid and handle midpoint for odd-length MPS
            (isodd(n) && i == n ÷ 2 + 1) ? (χl, χl) : (after_mid ? (χr, χl) : (χl, χr))
        end

        # orthogonalize by QR factorization
        F = lq!(rand(rng, T, χl, ip * op * χr))
        reshape(Matrix(F.Q), χl, ip, op, χr)
    end

    # reshape boundary sites
    arrays[1] = reshape(arrays[1], ip, op, min(χ, ip * op))
    arrays[n] = reshape(arrays[n], min(χ, ip * op), ip, op)

    # TODO order might not be the best for performance
    return MPO(arrays; order=(:l, :i, :o, :r))
end

# TODO deprecate contract(; between) and generalize it to AbstractAnsatz
"""
    Tenet.contract!(tn::AbstractMPO; between=(site1, site2), direction::Symbol = :left, delete_Λ = true)

For a given [`AbstractMPO`](@ref) Tensor Network, contract the singular values Λ between two sites `site1` and `site2`.
The `direction` keyword argument specifies the direction of the contraction, and the `delete_Λ` keyword argument
specifies whether to delete the singular values tensor after the contraction.
"""
@kwmethod contract(tn::AbstractMPO; between, direction, delete_Λ) = contract!(copy(tn); between, direction, delete_Λ)
@kwmethod function contract!(tn::AbstractMPO; between, direction, delete_Λ)
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
@kwmethod contract(tn::AbstractMPO; between) = contract(tn; between, direction=:left, delete_Λ=true)
@kwmethod contract!(tn::AbstractMPO; between) = contract!(tn; between, direction=:left, delete_Λ=true)
@kwmethod contract(tn::AbstractMPO; between, direction) = contract(tn; between, direction, delete_Λ=true)
@kwmethod contract!(tn::AbstractMPO; between, direction) = contract!(tn; between, direction, delete_Λ=true)

# TODO change it to `lanes`?
# TODO refactor to use `Lattice`
function sites(ψ::T, site::Site; dir) where {T<:AbstractMPO}
    if dir === :left
        return site <= site"1" ? nothing : Site(id(site) - 1)
    elseif dir === :right
        return site >= Site(nlanes(ψ)) ? nothing : Site(id(site) + 1)
    else
        throw(ArgumentError("Unknown direction for $T = :$dir"))
    end
end

# TODO refactor to use `Lattice`
@kwmethod function inds(ψ::T; at, dir) where {T<:AbstractMPO}
    if dir === :left && at == site"1"
        return nothing
    elseif dir === :right && at == Site(nlanes(ψ); dual=isdual(at))
        return nothing
    elseif dir ∈ (:left, :right)
        return inds(ψ; bond=(at, sites(ψ, at; dir)))
    else
        throw(ArgumentError("Unknown direction for $T = :$dir"))
    end
end

function isisometry(ψ::AbstractMPO, site; dir, atol::Real=1e-12)
    tensor = tensors(ψ; at=site)
    dirind = inds(ψ; at=site, dir)

    if isnothing(dirind)
        return isapprox(parent(contract(tensor, conj(tensor))), fill(true); atol)
    end

    inda, indb = gensym(:a), gensym(:b)
    a = replace(tensor, dirind => inda)
    b = replace(conj(tensor), dirind => indb)

    n = size(tensor, dirind)
    contracted = contract(a, b; out=[inda, indb])

    return isapprox(contracted, I(n); atol)
end

@deprecate isleftcanonical(ψ::AbstractMPO, site; atol::Real=1e-12) isisometry(ψ, site; dir=:right, atol)
@deprecate isrightcanonical(ψ::AbstractMPO, site; atol::Real=1e-12) isisometry(ψ, site; dir=:left, atol)

# TODO generalize to AbstractAnsatz
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

function canonize!(ψ::AbstractMPO)
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

    ψ.form = Canonical()

    return ψ
end

# TODO mixed_canonize! at bond
# TODO dispatch on form
# TODO generalize to AbstractAnsatz
function mixed_canonize!(tn::AbstractMPO, orthog_center)
    if orthog_center isa Site
        left = id(orthog_center) - 1
        right = id(orthog_center) + 1
    else
        values = [id(site) for site in orthog_center]
        orthog_center = Vector{Site}(orthog_center)

        left, right = extrema(values) .+ (-1, 1)
    end

    # left-to-right QR sweep (left-canonical tensors)
    for i in 1:left
        canonize_site!(tn, Site(i); direction=:right, method=:qr)
    end

    # right-to-left QR sweep (right-canonical tensors)
    for i in nsites(tn):-1:right
        canonize_site!(tn, Site(i); direction=:left, method=:qr)
    end

    # center SVD sweep to get singular values
    # for i in (left + 1):(right - 1)
    #     canonize_site!(tn, Site(i); direction=:left, method=:svd)
    # end

    tn.form = MixedCanonical(orthog_center)

    return tn
end

LinearAlgebra.normalize!(ψ::AbstractMPO; kwargs...) = normalize!(form(ψ), ψ; kwargs...)

function LinearAlgebra.normalize!(::NonCanonical, ψ::AbstractMPO; at=Site(nsites(ψ) ÷ 2))
    tensor = tensors(ψ; at)
    tensor ./= norm(ψ)
    return ψ
end

LinearAlgebra.normalize!(ψ::AbstractMPO, site::Site) = normalize!(mixed_canonize!(ψ, site); at=site)

function LinearAlgebra.normalize!(config::MixedCanonical, ψ::AbstractMPO; at=config.orthog_center)
    mixed_canonize!(ψ, at)
    normalize!(tensors(ψ; at), 2)
    return ψ
end

# TODO function LinearAlgebra.normalize!(::Canonical, ψ::AbstractMPO) end

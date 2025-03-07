using Random
using LinearAlgebra
using Graphs: Graphs

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
    lattice = Lattice(Val(:chain), n)
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

"""
    check_form(mps::AbstractMPO)

Check if the tensors in the mps are in the proper [`Form`](@ref).
"""
check_form(mps::AbstractMPO; kwargs...) = check_form(form(mps), mps; kwargs...)

function check_form(config::MixedCanonical, ψ::AbstractMPO; atol=1e-12)
    orthog_center = config.orthog_center

    left, right = if orthog_center isa Lane
        id(orthog_center) .+ (0, 0) # So left and right get the same value
    elseif orthog_center isa Vector{<:Lane}
        extrema(id.(orthog_center))
    end

    for i in 1:nlanes(ψ)
        if i < left # Check left-canonical tensors
            isisometry(ψ, Lane(i); dir=:right, atol) || throw(ArgumentError("Tensors are not left-canonical"))
        elseif i > right # Check right-canonical tensors
            isisometry(ψ, Lane(i); dir=:left, atol) || throw(ArgumentError("Tensors are not right-canonical"))
        end
    end

    return true
end

function check_form(::Canonical, mps::AbstractMPO; atol=1e-12)
    for i in 1:nlanes(mps)
        if i > 1 && !isisometry(absorb(mps; bond=(Lane(i - 1), Lane(i)), dir=:right), Lane(i); dir=:right, atol)
            throw(ArgumentError("Can not form a left-canonical tensor in Lane($i) from Γ and λ contraction."))
        end

        if i < nlanes(mps) && !isisometry(absorb(mps; bond=(Lane(i), Lane(i + 1)), dir=:left), Lane(i); dir=:left, atol)
            throw(ArgumentError("Can not form a right-canonical tensor in Site($i) from Γ and λ contraction."))
        end
    end

    return true
end

check_form(::NonCanonical, mps::AbstractMPO) = true

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
    lattice = Lattice(Val(:chain), n)
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
  - `maxdim` The maximum bond dimension.  If it is `nothing`, the maximum bond dimension increases exponentially with the number of sites up to `physdim^(n ÷ 2)`.
  - `eltype` The element type of the tensors. Defaults to `Float64`.
  - `physdim` The physical or output dimension of each site. Defaults to 2.
"""
function Base.rand(rng::Random.AbstractRNG, ::Type{MPS}; n, maxdim=nothing, eltype=Float64, physdim=2)
    p = physdim
    T = eltype
    χ = isnothing(maxdim) ? p^(n ÷ 2) : maxdim

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

    return MPS(arrays; order=(:l, :o, :r), form=MixedCanonical(Lane(1)))
end

# TODO different input/output physical dims
# TODO let choose the orthogonality center
"""
    Base.rand(rng::Random.AbstractRNG, ::Type{MPO}; n, maxdim, eltype=Float64, physdim=2)

Create a random [`MPO`](@ref) Tensor Network.
In order to avoid norm explosion issues, the tensors are orthogonalized by QR factorization so its normalized and mixed canonized to the last site.

# Keyword Arguments

  - `n` The number of sites.
  - `maxdim` The maximum bond dimension. If it is `nothing`, the maximum bond dimension increases exponentially with the number of sites up to `(physdim^2)^(n ÷ 2)`.
  - `eltype` The element type of the tensors. Defaults to `Float64`.
  - `physdim` The physical or output dimension of each site. Defaults to 2.
"""
function Base.rand(rng::Random.AbstractRNG, ::Type{MPO}; n, maxdim=nothing, eltype=Float64, physdim=2)
    T = eltype
    ip = op = physdim
    χ = isnothing(maxdim) ? (ip * op)^(n ÷ 2) : maxdim

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

# TODO generalize it to AbstractAnsatz
# TODO instead of `delete_Λ`, make another function for the reduced density matrix
"""
    absorb!(tn::AbstractMPO; bond=(lane1, lane2), dir::Symbol = :left, delete_Λ = true)

For a given [`AbstractMPO`](@ref) Tensor Network, contract the singular values Λ located in the bond between lanes `lane1` and `lane2`.

# Keyword arguments

    - `bond` The bond between the singular values tensor and the tensors to be contracted.
    - `dir` The direction of the contraction. Defaults to `:left`.
    - `delete_Λ` Whether to delete the singular values tensor after the contraction. Defaults to `true`.
"""
function absorb!(tn::AbstractMPO; bond, delete_Λ=true, dir=:left)
    lane1, lane2 = bond
    Λᵢ = tensors(tn; bond=bond)
    isnothing(Λᵢ) && return tn

    if dir === :right
        Γᵢ₊₁ = tensors(tn; at=lane2)
        replace!(tn, Γᵢ₊₁ => contract(Γᵢ₊₁, Λᵢ; dims=()))
    elseif dir === :left
        Γᵢ = tensors(tn; at=lane1)
        replace!(tn, Γᵢ => contract(Λᵢ, Γᵢ; dims=()))
    else
        throw(ArgumentError("Unknown direction=:$(dir)"))
    end

    delete_Λ && delete!(TensorNetwork(tn), Λᵢ)

    return tn
end

"""
    absorb(tn::AbstractMPO; kwargs...)

Non-mutating version of [`absorb!`](@ref).
"""
absorb(tn::AbstractMPO; kwargs...) = absorb!(copy(tn); kwargs...)

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

function lanes(ψ::T, lane::Lane; dir) where {T<:AbstractMPO}
    if dir === :left
        return lane <= lane"1" ? nothing : Lane(id(lane) - 1)
    elseif dir === :right
        return lane >= Lane(nlanes(ψ)) ? nothing : Lane(id(lane) + 1)
    else
        throw(ArgumentError("Unknown direction for $T = :$dir"))
    end
end

# TODO refactor to use `Lattice`
function inds(kwargs::NamedTuple{(:at, :dir),Tuple{S,Symbol}}, ψ::T) where {S<:Site,T<:AbstractMPO}
    if kwargs.dir === :left && kwargs.at == site"1"
        return nothing
    elseif kwargs.dir === :right && kwargs.at == Site(nlanes(ψ); dual=isdual(kwargs.at))
        return nothing
    elseif kwargs.dir ∈ (:left, :right)
        return inds(ψ; bond=(kwargs.at, sites(ψ, kwargs.at; dir=kwargs.dir)))
    else
        throw(ArgumentError("Unknown direction for $T = :$(kwargs.dir)"))
    end
end

function inds(kwargs::NamedTuple{(:at, :dir),Tuple{L,Symbol}}, ψ::T) where {L<:Lane,T<:AbstractMPO}
    if kwargs.dir === :left && kwargs.at == lane"1"
        return nothing
    elseif kwargs.dir === :right && kwargs.at == Lane(nlanes(ψ))
        return nothing
    elseif kwargs.dir ∈ (:left, :right)
        return inds(ψ; bond=(kwargs.at, lanes(ψ, kwargs.at; dir=kwargs.dir)))
    else
        throw(ArgumentError("Unknown direction for $T = :$(kwargs.dir)"))
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

@deprecate isleftcanonical(ψ::AbstractMPO, lane; atol::Real=1e-12) isisometry(ψ, lane; dir=:right, atol)
@deprecate isrightcanonical(ψ::AbstractMPO, lane; atol::Real=1e-12) isisometry(ψ, lane; dir=:left, atol)

# TODO generalize to AbstractAnsatz
# NOTE: in method == :svd the spectral weights are stored in a vector connected to the now virtual hyperindex!
function canonize_site!(ψ::AbstractMPO, lane::Lane; dir::Symbol, method=:qr)
    left_inds = Symbol[]
    right_inds = Symbol[]
    site = Site(lane)

    virtualind = if dir === :left
        lane == lane"1" && throw(ArgumentError("Cannot right-canonize left-most tensor"))
        push!(right_inds, inds(ψ; at=lane, dir=:left))

        lane == Lane(nlanes(ψ)) || push!(left_inds, inds(ψ; at=lane, dir=:right))
        site ∈ ψ && push!(left_inds, inds(ψ; at=site))
        site' ∈ ψ && push!(left_inds, inds(ψ; at=site'))

        only(right_inds)
    elseif dir === :right
        lane == Lane(nlanes(ψ)) && throw(ArgumentError("Cannot left-canonize right-most tensor"))
        push!(right_inds, inds(ψ; at=lane, dir=:right))

        lane == lane"1" || push!(left_inds, inds(ψ; at=lane, dir=:left))
        site ∈ ψ && push!(left_inds, inds(ψ; at=site))
        site' ∈ ψ && push!(left_inds, inds(ψ; at=site'))

        only(right_inds)
    else
        throw(ArgumentError("Unknown direction=:$dir"))
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

function canonize!(ψ::AbstractMPO; normalize=false)
    Λ = Tensor[]

    # right-to-left QR sweep, get right-canonical tensors
    for i in nlanes(ψ):-1:2
        canonize_site!(ψ, Lane(i); dir=:left, method=:qr)
    end

    # left-to-right SVD sweep, get left-canonical tensors and singular values without reversing
    for i in 1:(nlanes(ψ) - 1)
        canonize_site!(ψ, Lane(i); dir=:right, method=:svd)

        # extract the singular values and contract them with the next tensor
        Λᵢ = pop!(ψ, tensors(ψ; bond=(Lane(i), Lane(i + 1))))
        normalize && (Λᵢ ./= norm(Λᵢ))
        Aᵢ₊₁ = tensors(ψ; at=Lane(i + 1))
        replace!(ψ, Aᵢ₊₁ => contract(Aᵢ₊₁, Λᵢ; dims=()))
        push!(Λ, Λᵢ)
    end

    for i in 2:nlanes(ψ) # tensors at i in "A" form, need to contract (Λᵢ)⁻¹ with A to get Γᵢ
        Λᵢ = Λ[i - 1] # singular values start between lane 1 and 2
        A = tensors(ψ; at=Lane(i))
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
    left, right = if orthog_center isa Lane
        id(orthog_center) .+ (-1, 1)
    elseif orthog_center isa Vector{<:Lane}
        extrema(id.(orthog_center)) .+ (-1, 1)
    else
        throw(ArgumentError("`orthog_center` must be a `Site` or a `Vector{Site}`"))
    end

    # left-to-right QR sweep (left-canonical tensors)
    for i in 1:left
        canonize_site!(tn, Lane(i); dir=:right, method=:qr)
    end

    # right-to-left QR sweep (right-canonical tensors)
    for i in nlanes(tn):-1:right
        canonize_site!(tn, Lane(i); dir=:left, method=:qr)
    end

    tn.form = MixedCanonical(orthog_center)

    return tn
end

"""
    evolve!(ψ::AbstractAnsatz, mpo::AbstractMPO; threshold=nothing, maxdim=nothing, normalize=true, reset_index=true)

Evolve the [`AbstractAnsatz`](@ref) `ψ` with the [`AbstractMPO`](@ref) `mpo` along the output indices of `ψ`.
If `threshold` or `maxdim` are not `nothing`, the tensors are truncated after each sweep at the proper value, and the
bond is normalized if `normalize=true`. If `reset_index=true`, the indices of the `ψ` are reset to the original ones.
"""
function evolve!(ψ::AbstractMPS, mpo::AbstractMPO; reset_index=true, kwargs...)
    original_sites = copy(Quantum(ψ).sites)
    normalize = get(kwargs, :normalize, true)
    evolve!(form(ψ), ψ, mpo; normalize, kwargs...)

    if reset_index
        resetinds!(ψ; init=ninds(TensorNetwork(ψ)) + 1)

        replacements = [inds(ψ; at=site) => original_sites[site] for site in keys(original_sites)]
        replace!(ψ, replacements)
    end

    return ψ
end

function evolve!(::NonCanonical, ψ::AbstractMPS, H::AbstractMPO; kwargs...)
    @assert nlanes(ψ) == nlanes(H)

    # align but don't merge to extract information
    Tenet.@reindex! outputs(ψ) => inputs(H)
    bond_inds = [inds(ψ; bond=(Lane(i), Lane(i + 1))) for i in 1:(nlanes(ψ) - 1)]
    phys_inds = [inds(ψ; at=site) for site in sites(ψ; set=:outputs)]

    # merge and contract inner physical indices
    merge!(Quantum(ψ), Quantum(H); reset=false)

    for ind in phys_inds
        contract!(ψ, ind)
    end

    # group the parallel bond indices
    for ind in bond_inds
        fuse!(ψ, ind)
    end

    # NOTE `fuse!(::AbstractTensorNetwork)`` calls `pop!` inside so we must relink sites to inds
    # TODO fix this on interface refactor
    for site in sites(H; set=:outputs)
        addsite!(ψ, site, inds(H; at=site))
    end

    truncate_sweep!(form(ψ), ψ; kwargs...)

    if all(isnothing, get.(Ref(kwargs), [:threshold, :maxdim], nothing))
        normalize = get(kwargs, :normalize, true)
        normalize && normalize!(ψ)
    end

    return ψ
end

function evolve!(::MixedCanonical, ψ::AbstractMPS, mpo::AbstractMPO; kwargs...)
    initial_form = form(ψ)
    mixed_canonize!(ψ, Lane(nlanes(ψ))) # We convert all the tensors to left-canonical form

    normalize = get(kwargs, :normalize, true)
    evolve!(NonCanonical(), ψ, mpo; normalize, kwargs...)

    mixed_canonize!(ψ, initial_form.orthog_center)

    return ψ
end

function evolve!(::Canonical, ψ::AbstractMPS, mpo::AbstractMPO; kwargs...)
    # We first join the λs to the Γs to get MixedCanonical(lane"1") form
    for i in 1:(nlanes(ψ) - 1)
        absorb!(ψ; bond=(Lane(i), Lane(i + 1)), dir=:right)
    end

    # set `maxdim` and `threshold` to `nothing` so we later truncate in the `Canonical` form
    evolve!(NonCanonical(), ψ, mpo; kwargs..., threshold=nothing, maxdim=nothing, normalize=false)
    truncate_sweep!(Canonical(), ψ; kwargs...)

    if all(isnothing, get.(Ref(kwargs), [:threshold, :maxdim], nothing))
        normalize = get(kwargs, :normalize, true)
        normalize && canonize!(ψ; normalize)
    end

    return ψ
end

"""
    truncate_sweep!

Do a right-to-left QR sweep on the [`AbstractMPO`](@ref) `ψ` and then left-to-right SVD sweep and truncate the tensors
according to the `threshold` or `maxdim` values. The bond is normalized if `normalize=true`.
"""
truncate_sweep!(ψ::AbstractMPO; kwargs...) = truncate_sweep!(form(ψ), ψ; kwargs...)

function truncate_sweep!(::NonCanonical, ψ::AbstractMPO; kwargs...)
    all(isnothing, get.(Ref(kwargs), [:threshold, :maxdim], nothing)) && return ψ

    for i in nlanes(ψ):-1:2
        canonize_site!(ψ, Lane(i); dir=:left, method=:qr)
    end

    # left-to-right SVD sweep, get left-canonical tensors and singular values and truncate
    for i in 1:(nlanes(ψ) - 1)
        canonize_site!(ψ, Lane(i); dir=:right, method=:svd)

        truncate!(ψ, [Lane(i), Lane(i + 1)]; kwargs..., compute_local_svd=false)
        absorb!(ψ; bond=(Lane(i), Lane(i + 1)), dir=:right)
    end

    ψ.form = MixedCanonical(Lane(nlanes(ψ)))

    return ψ
end

function truncate_sweep!(::MixedCanonical, ψ::AbstractMPO; kwargs...)
    truncate_sweep!(NonCanonical(), ψ; kwargs...)
end

function truncate_sweep!(::Canonical, ψ::AbstractMPO; kwargs...)
    all(isnothing, get.(Ref(kwargs), [:threshold, :maxdim], nothing)) && return ψ

    for i in nlanes(ψ):-1:2
        canonize_site!(ψ, Lane(i); dir=:left, method=:qr)
    end

    # left-to-right SVD sweep, get left-canonical tensors and singular values and truncate
    for i in 1:(nlanes(ψ) - 1)
        canonize_site!(ψ, Lane(i); dir=:right, method=:svd)
        truncate!(ψ, [Lane(i), Lane(i + 1)]; kwargs..., compute_local_svd=false)
    end

    canonize!(ψ)

    return ψ
end

LinearAlgebra.normalize!(ψ::AbstractMPO; kwargs...) = normalize!(form(ψ), ψ; kwargs...)
LinearAlgebra.normalize!(ψ::AbstractMPO, at::Lane) = normalize!(form(ψ), ψ; at)
LinearAlgebra.normalize!(ψ::AbstractMPO, bond::Base.AbstractVecOrTuple{Lane}) = normalize!(form(ψ), ψ; bond)

# NOTE: Inplace normalization of the arrays should be faster, but currently lead to problems for `copy` TensorNetworks
function LinearAlgebra.normalize!(::NonCanonical, ψ::AbstractMPO; at=Lane(nlanes(ψ) ÷ 2))
    if at isa Site
        tensor = tensors(ψ; at)
        replace!(ψ, tensor => tensor ./ norm(ψ))
    else
        normalize!(mixed_canonize!(ψ, at))
    end

    return ψ
end

function LinearAlgebra.normalize!(config::MixedCanonical, ψ::AbstractMPO; at=config.orthog_center)
    mixed_canonize!(ψ, at)
    normalize!(tensors(ψ; at), 2)
    return ψ
end

function LinearAlgebra.normalize!(::Canonical, ψ::AbstractMPO; bond=nothing)
    if !isnothing(bond)
        # when setting `bond`, we are just normalizing one Λ tensor and its neighbor Γ tensors
        Λab = tensors(ψ; bond)
        normalize!(Λab)

        a, b = bond
        Γa, Γb = tensors(ψ; at=a), tensors(ψ; at=b)

        # γ are Γ tensors with neighbor Λ tensors contracted => γ = Λ Γ Λ
        # i.e. it's half reduced density matrix for the site, so it's norm is the total norm too
        # NOTE this works only if the state is correctly canonized!
        γa, γb = contract(Γa, Λab; dims=Symbol[]), contract(Γb, Λab; dims=Symbol[])

        # open boundary conditions
        if a != lane"1"
            Λa = tensors(ψ; bond=(Lane(id(a - 1)), a))
            γa = contract(γa, Λa; dims=Symbol[])
        end

        if b != Lane(nlanes(ψ))
            Λb = tensors(ψ; bond=(b, Lane(id(b + 1))))
            γb = contract(γb, Λb; dims=Symbol[])
        end

        Za, Zb = norm(γa), norm(γb)

        Γa ./= Za
        Γb ./= Zb
    end

    # normalize the Λ tensors
    for i in 1:(nlanes(ψ) - 1)
        Λ = tensors(ψ; bond=(Lane(i), Lane(i + 1)))
        normalize!(Λ)
    end

    # normalize the Γ tensors
    for i in 2:(nlanes(ψ) - 1)
        Γ = tensors(ψ; at=Lane(i))
        Λᵢ₋₁ = tensors(ψ; bond=(Lane(i - 1), Lane(i)))
        Λᵢ₊₁ = tensors(ψ; bond=(Lane(i), Lane(i + 1)))

        # NOTE manual binary contraction due to bugs in `contract(args...)`
        Z = norm(contract(contract(Γ, Λᵢ₋₁; dims=Symbol[]), Λᵢ₊₁; dims=Symbol[]))
        Γ ./= Z
    end

    return ψ
end

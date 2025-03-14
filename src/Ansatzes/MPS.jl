using Random
using LinearAlgebra
using Graphs: Graphs

abstract type AbstractMPO <: AbstractTensorNetwork end
abstract type AbstractMPS <: AbstractMPO end

"""
    MPS <: AbstractMPS

A Matrix Product State [`Ansatz`](@ref) Tensor Network.
"""
mutable struct MatrixProductState <: AbstractMPS
    const tn::TensorNetwork
    const pluggable::PluggableMixin
    const ansatz::AnsatzMixin
    form::Form
end

const MPS = MatrixProductState

"""
    MPO <: AbstractMPO

A Matrix Product Operator (MPO) [`Ansatz`](@ref) Tensor Network.
"""
mutable struct MatrixProductOperator <: AbstractMPO
    const tn::TensorNetwork
    const pluggable::PluggableMixin
    const ansatz::AnsatzMixin
    form::Form
end

const MPO = MatrixProductOperator

# mutable struct MPDO <: AbstractMPO
#     const tn::TensorNetwork
#     const pluggable::Pluggable
#     const ansatz::AnsatzMixin
#     form::Form
# end

boundary(::Union{MPS,MPO}) = Open()
form(tn::Union{MPS,MPO}) = tn.form
lattice(tn::Union{MPS,MPO}) = tn.lattice

# function Base.deepcopy(tn::T) where {T<:Union{MPS,MPO}}
#     tn = deepcopy(tn)
# end

# function Base.similar(tn::T) where {T<:Union{MPS,MPO}}
#     T(similar(TensorNetwork(tn)), lattice(tn), copy(tn.lanemap), copy(tn.bondmap), copy(tn.sitemap), form(tn))
# end

# function Base.zero(tn::T) where {T<:Union{MPS,MPO}}
#     T(zero(TensorNetwork(tn)), lattice(tn), copy(tn.lanemap), copy(tn.bondmap), copy(tn.sitemap), form(tn))
# end

defaultorder(::Type{<:AbstractMPS}) = (:o, :l, :r)
defaultorder(::Type{<:AbstractMPO}) = (:o, :i, :l, :r)

MPS(arrays; form::Form=NonCanonical(), kwargs...) = MPS(form, arrays; kwargs...)
function MPS(arrays::Vector{<:AbstractArray}, λ; kwargs...)
    return MPS(Canonical(), arrays, λ; kwargs...)
end

# Tensor Network interface
trait(::TensorNetworkInterface, ::Union{MPS,MPO}) = WrapsTensorNetwork()
unwrap(::TensorNetworkInterface, tn::Union{MPS,MPO}) = tn.tn

function Base.copy(tn::T) where {T<:Union{MPS,MPO}}
    T(copy(tn.tn), copy(tn.pluggable), copy(tn.ansatz), form(tn))
end

# Pluggable interface
trait(::PluggableInterface, ::Union{MPS,MPO}) = WrapsPluggable()
unwrap(::PluggableInterface, tn::Union{MPS,MPO}) = tn.pluggable

# Ansatz interface
trait(::AnsatzInterface, ::Union{MPS,MPO}) = WrapsAnsatz()
unwrap(::AnsatzInterface, tn::Union{MPS,MPO}) = tn.ansatz

# effect handlers
function handle!(tn::Union{MPS,MPO}, effect::ReplaceEffect{Pair{Symbol,Symbol}})
    handle!(unwrap(PluggableInterface(), tn), effect)
end

# TODO should we add a flag to check if the tensor fulfills the canonical form?
function handle!(tn::Union{MPS,MPO}, effect::ReplaceEffect{Pair{Tensor,Tensor}})
    handle!(unwrap(AnsatzInterface(), tn), effect)
end

# constructors
"""
    checkform(mps::AbstractMPO)

Check if the tensors in the mps are in the proper [`Form`](@ref).
"""
checkform(tn::AbstractMPO; kwargs...) = checkform(form(tn), tn; kwargs...)

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
    lattice = Lattice(Val(:chain), n)

    sitemap = Dict{Site,Symbol}(Site(i) => nextindex!(gen) for i in 1:n)
    bondmap = Dict{Bond,Symbol}(bond => nextindex!(gen) for bond in Graphs.edges(lattice))
    lanemap = Dict{Lane,Tensor}(
        map(enumerate(arrays)) do (i, array)
            local_order = if i == 1
                filter(x -> x != :l, order)
            elseif i == n
                filter(x -> x != :r, order)
            else
                order
            end

            inds = map(local_order) do dir
                if dir == :o
                    sitemap[Site(i)]
                elseif dir == :r
                    bondmap[Bond(Lane(i), Lane(i + 1))]
                elseif dir == :l
                    bondmap[Bond(Lane(i - 1), Lane(i))]
                else
                    throw(ArgumentError("Invalid direction: $dir"))
                end
            end

            Lane(i) => Tensor(array, inds)
        end,
    )

    tn = TensorNetwork(values(lanemap))
    pluggable = PluggableMixin(sitemap)
    ansatz = AnsatzMixin(lanemap, bondmap)

    return MPS(tn, pluggable, ansatz, NonCanonical())
end

checkform(::NonCanonical, mps::AbstractMPO) = true

# TODO should we set `check=false` for performance?
"""
    MPS(MixedCanonical(), arrays; order=defaultorder(MPS), check=true)

Create a [`MixedCanonical`](@ref) [`MPS`](@ref) from a vector of arrays.

# Keyword Arguments

  - `order` The order of the indices in the arrays. Defaults to `(:o, :l, :r)`.
  - `check` Whether to check the canonical form of the MPS.
"""
function MPS(form::MixedCanonical, arrays; order=defaultorder(MPS), check=true)
    mps = MPS(arrays; form=NonCanonical(), order, check)
    mps.form = form
    check && checkform(mps)
    return mps
end

function checkform(config::MixedCanonical, tn::AbstractMPO; atol=1e-11)
    orthog_center = config.orthog_center

    left, right = if orthog_center isa Lane
        id(orthog_center) .+ (0, 0) # So left and right get the same value
    elseif orthog_center isa Vector{<:Lane}
        extrema(id.(orthog_center))
    end

    for i in 1:nlanes(tn)
        if i < left # Check left-canonical tensors
            isisometry(tn, Lane(i); dir=:right, atol) ||
                throw(ArgumentError("Tensors to the left of lane $i are not left-canonical"))
        elseif i > right # Check right-canonical tensors
            isisometry(tn, Lane(i); dir=:left, atol) ||
                throw(ArgumentError("Tensors to the right of lane $i are not right-canonical"))
        end
    end

    return true
end

"""
    MPS(Canonical(), Γ, λ; order=defaultorder(MPS), check=true)

Create a [`Canonical`](@ref) [`MPS`](@ref) from a vector of arrays.

# Keyword Arguments

  - `order` The order of the indices in the arrays. Defaults to `(:o, :l, :r)`.
  - `check` Whether to check the canonical form of the MPS.
"""
function MPS(::Canonical, Γ, λ; order=defaultorder(MPS), check=true)
    @assert length(λ) == length(Γ) - 1 "Number of λ tensors must be one less than the number of Γ tensors"
    @assert all(==(1) ∘ ndims, λ) "All λ tensors must be 1-dimensional"

    mps = MPS(Γ; form=NonCanonical(), order, check)
    mps.form = Canonical()

    # create tensors from 'λ'
    map(enumerate(λ)) do (i, array)
        bondind = inds(mps; bond=Bond(Lane(i), Lane(i + 1)))
        tensor = Tensor(array, (bondind,))
        push_inner!(mps, tensor)
    end

    # check canonical form by contracting Γ and λ tensors and checking their orthogonality
    check && checkform(mps)

    return mps
end

function checkform(::Canonical, tn::AbstractMPO; atol=1e-11)
    for i in 1:nlanes(tn)
        if i > 1 && !isisometry(absorb(tn; bond=(Lane(i - 1), Lane(i)), dir=:right), Lane(i); dir=:right, atol)
            throw(ArgumentError("Can not form a left-canonical tensor in lane $i from Γ and λ contraction"))
        end

        if i < nlanes(tn) && !isisometry(absorb(tn; bond=(Lane(i), Lane(i + 1)), dir=:left), Lane(i); dir=:left, atol)
            throw(ArgumentError("Can not form a right-canonical tensor in lane $i from Γ and λ contraction"))
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
    lattice = Lattice(Val(:chain), n)

    sitemap = Dict{Site,Symbol}(Site(i) => nextindex!(gen) for i in 1:n)
    append!(sitemap, [Site(i; dual=true) => nextindex!(gen) for i in 1:n])
    bondmap = Dict{Bond,Symbol}(bond => nextindex!(gen) for bond in Graphs.edges(lattice))

    lanemap = Dict{Lane,Tensor}(
        map(enumerate(arrays)) do (i, array)
            local_order = if i == 1
                filter(x -> x != :l, order)
            elseif i == n
                filter(x -> x != :r, order)
            else
                order
            end

            inds = map(local_order) do dir
                if dir == :o
                    sitemap[Site(i)]
                elseif dir == :i
                    sitemap[Site(i; dual=true)]
                elseif dir == :r
                    bondmap[Bond(Lane(i), Lane(i + 1))]
                elseif dir == :l
                    bondmap[Bond(Lane(i - 1), Lane(i))]
                else
                    throw(ArgumentError("Invalid direction: $dir"))
                end
            end

            Lane(i) => Tensor(array, inds)
        end,
    )

    tn = TensorNetwork(values(lanemap))
    pluggable = PluggableMixin(sitemap)
    ansatz = AnsatzMixin(lanemap, bondmap)

    return MPO(tn, pluggable, ansatz, NonCanonical())
end

################################################################################
# TODO normalize as we canonize for numerical stability
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

# TODO normalize as we canonize for numerical stability
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

# TODO canonization methods: canonize!, canonize_site!, absorb!, ...
# TODO improve over `evolve!` methods?
# TODO improve over `truncate!` methods?

# derived methods
LinearAlgebra.norm(ψ::AbstractMPO) = norm(form(ψ), tensors(ψ))

function LinearAlgebra.norm(::NonCanonical, tn)
    # TODO stack with its dual and contract
    error("Not implemented yet")
end

function LinearAlgebra.norm(config::MixedCanonical, tn)
    orthog_center = tensors(tn; at=config.orthog_center)
    return norm(orthog_center)
end

function LinearAlgebra.norm(::Canonical, tn)
    # TODO should we just return the norm of one of the Λ tensors? take an average for numerical stability?
    error("Not implemented yet")
end

LinearAlgebra.normalize!(ψ::AbstractMPO; kwargs...) = normalize!(form(ψ), ψ; kwargs...)
LinearAlgebra.normalize!(ψ::AbstractMPO, at::Lane) = normalize!(form(ψ), ψ; at)
LinearAlgebra.normalize!(ψ::AbstractMPO, bond::Base.AbstractVecOrTuple{Lane}) = normalize!(form(ψ), ψ; bond)

# NOTE in-place normalization of the arrays should be faster, but currently leads to problems for `copy` TensorNetworks
function LinearAlgebra.normalize!(::NonCanonical, ψ::AbstractMPO; at=nothing)
    if isnothing(at)
        spread_norm = norm(ψ)^(1 / ntensors(ψ))
        for tensor in tensors(ψ)
            tensor ./= spread_norm
        end
    else
        tensor = tensors(ψ; at)
        replace!(ψ, tensor => tensor ./ norm(ψ))
    end
    return ψ
end

function LinearAlgebra.normalize!(config::MixedCanonical, ψ::AbstractMPO; at=config.orthog_center)
    # moves orthogonality center to the specified lane (does nothing if already there)
    canonize!(ψ, MixedCanonical(at))

    # orthogonality center contains all the norm, so just normalize that tensor
    normalize!(tensors(ψ; at), 2)

    return ψ
end

function LinearAlgebra.normalize!(::Canonical, ψ::AbstractMPO; bond=nothing)
    old_norm = norm(ψ)
    if isnothing(bond) # Normalize all λ tensors
        for i in 1:(nlanes(ψ) - 1)
            λ = tensors(ψ; bond=(Lane(i), Lane(i + 1)))
            replace!(ψ, λ => λ ./ old_norm^(1 / (nlanes(ψ) - 1)))
        end
    else
        λ = tensors(ψ; bond)
        replace!(ψ, λ => λ ./ old_norm)
    end

    return ψ
end

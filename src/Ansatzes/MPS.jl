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
            isisometry(tn, Lane(i), :right; atol) ||
                throw(ArgumentError("Tensors to the left of lane $i are not left-canonical"))
        elseif i > right # Check right-canonical tensors
            isisometry(tn, Lane(i), :left; atol) ||
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
    # boundary conditions
    if !isisometry(tn, lane"1", Bond(lane"1", lane"2"); atol)
        throw(AssertionError("Tensor at lane 1 is not an isommetry"))
    end

    if !isisometry(tn, Lane(nlanes(tn)), Bond(Lane(nlanes(tn) - 1), Lane(nlanes(tn))); atol)
        throw(AssertionError("Tensor at lane $(nlanes(tn)) is not an isommetry"))
    end

    # check canonical form by contracting Γ and λ tensors and checking their orthogonality
    for i in 2:(nlanes(tn) - 1)
        Λleft = tensors(tn; bond=Bond(Lane(i - 1), Lane(i)))
        leftind = inds(tn; bond=Bond(Lane(i - 1), Lane(i)))

        Λright = tensors(tn; bond=Bond(Lane(i), Lane(i + 1)))
        rightind = inds(tn; bond=Bond(Lane(i), Lane(i + 1)))
        tensor = tensors(tn; at=Lane(i))

        # absorbing left Λ tensor forms a isometry to the right (left-canonical tensor)
        isoright = contract(Λleft, tensor; dims=())
        if !isisometry(isoright, rightind; atol)
            throw(AssertionError("Can not form a left-canonical tensor in lane $i from Γ and λ contraction"))
        end

        # absorbing right Λ tensor forms a isometry to the left (right-canonical tensor)
        isoleft = contract(Λright, tensor; dims=())
        if !isisometry(isoleft, leftind; atol)
            throw(AssertionError("Can not form a right-canonical tensor in lane $i from Γ and λ contraction"))
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

"""
    lanes(ψ::T, lane::Lane; dir)

Return the [`Lane`](@ref) next to `lane` in the specified direction.
"""
function lanes(ψ::T, lane::Lane; dir) where {T<:AbstractMPO}
    if dir === :left
        return lane <= lane"1" ? nothing : Lane(id(lane) - 1)
    elseif dir === :right
        return lane >= Lane(nlanes(ψ)) ? nothing : Lane(id(lane) + 1)
    else
        throw(ArgumentError("Unknown direction for $T = :$dir"))
    end
end

"""
    bonds(ψ::AbstractMPO, lane::Lane; dir)

Keyword-dispatch variant of `bonds(tn, lane)` where only the [`Bond`](@ref) in the specified direction is returned.
"""
function bonds(kwargs::@NamedTuple{dir::Symbol}, ψ::AbstractMPO, lane::Lane)
    if kwargs.dir === :left
        return lane <= lane"1" ? nothing : Bond(Lane(id(lane) - 1), lane)
    elseif kwargs.dir === :right
        return lane >= Lane(nlanes(ψ)) ? nothing : Bond(lane, Lane(id(lane) + 1))
    else
        throw(ArgumentError("Unknown direction for $T = :$(kwargs.dir)"))
    end
end

# TODO refactor to use `bonds`
function inds(kwargs::NamedTuple{(:at, :dir),Tuple{L,Symbol}}, ψ::T) where {L<:Lane,T<:AbstractMPO}
    if kwargs.dir === :left && kwargs.at == lane"1"
        return nothing
    elseif kwargs.dir === :right && kwargs.at == Lane(nlanes(ψ))
        return nothing
    elseif kwargs.dir ∈ (:left, :right)
        return inds(ψ; bond=Bond(kwargs.at, lanes(ψ, kwargs.at; dir=kwargs.dir)))
    else
        throw(ArgumentError("Unknown direction for $T = :$(kwargs.dir)"))
    end
end

function isisometry(ψ::T, lane::Lane, dir::Symbol; kwargs...) where {T<:AbstractMPO}
    if dir === :left
        bond = Bond(Lane(id(lane) - 1), lane)
    elseif dir === :right
        bond = Bond(lane, Lane(id(lane) + 1))
    else
        throw(ArgumentError("Unknown direction for $T = :$dir"))
    end
    return isisometry(ψ, lane, bond; kwargs...)
end

# derived methods
# aliases for named directions
function canonize_site!(ψ::AbstractMPO, lane::Lane; dir::Symbol, kwargs...)
    @assert haslane(ψ, lane) "Lane $lane not found"
    bond = bonds(ψ, lane; dir)
    isnothing(bond) && throw(ArgumentError("There is no bond on $lane in direction $dir"))
    canonize_site!(ψ, lane, bond; kwargs...)
end

function canonize_site(ψ::AbstractMPO, lane::Lane; dir::Symbol, kwargs...)
    @assert haslane(ψ, lane) "Lane $lane not found"
    bond = bonds(ψ, lane; dir)
    isnothing(bond) && throw(ArgumentError("There is no bond on $lane in direction $dir"))
    canonize_site(ψ, lane, bond; kwargs...)
end

# TODO mixed_canonize! at bond
canonize!(tn::AbstractMPO, targetform::Form; kwargs...) = canonize!(form(tn), tn, targetform; kwargs...)
function canonize!(tn::AbstractMPO, lane::Union{L,Vector{L}}; kwargs...) where {L<:Lane}
    canonize!(form(tn), tn, MixedCanonical(lane); kwargs...)
end
canonize!(tn::AbstractMPO; kwargs...) = canonize!(form(tn), tn, Canonical(); kwargs...)

function canonize!(::Form, tn::AbstractMPO, ::NonCanonical)
    tn.form = NonCanonical()
    return tn
end

function canonize!(::NonCanonical, tn::AbstractMPO, targetform::MixedCanonical)
    canonize!(MixedCanonical(Lane.(1:nlanes(tn))), tn, targetform)
end

function canonize!(::Canonical, tn::AbstractMPO, targetform::MixedCanonical; sweep=true)
    min_orthog_center, max_orthog_center = if targetform.orthog_center isa Lane
        targetform.orthog_center, targetform.orthog_center
    elseif targetform.orthog_center isa Vector{<:Lane}
        extrema(id.(targetform.orthog_center))
    else
        throw(ArgumentError("`orthog_center` must be a `Lane` or a `Vector{Lane}`"))
    end

    for i in 1:(id(min_orthog_center) - 1)
        bond = Bond(Lane(i), Lane(i + 1))
        absorb!(tn, bond, :right)
    end

    for i in nlanes(tn):-1:(id(max_orthog_center) + 1)
        bond = Bond(Lane(i - 1), Lane(i))
        absorb!(tn, bond, :left)
    end

    # a sweep is need to fully propagate the effects of truncation
    # TODO probably there is a better way to propagate these effects
    sweep && canonize!(NonCanonical(), tn, targetform)

    tn.form = targetform

    return tn
end

function canonize!(srcform::MixedCanonical, tn, dstform::MixedCanonical)
    src_orthog_center = srcform.orthog_center
    src_left, src_right = if src_orthog_center isa Lane
        id(src_orthog_center), id(src_orthog_center)
    elseif src_orthog_center isa Vector{<:Lane}
        extrema(id.(src_orthog_center))
    else
        throw(ArgumentError("`orthog_center` must be a `Lane` or a `Vector{Lane}`"))
    end

    dst_orthog_center = dstform.orthog_center
    dst_left, dst_right = if dst_orthog_center isa Lane
        id(dst_orthog_center) .+ (-1, 1)
    elseif dst_orthog_center isa Vector{<:Lane}
        extrema(id.(dst_orthog_center)) .+ (-1, 1)
    else
        throw(ArgumentError("`orthog_center` must be a `Lane` or a `Vector{Lane}`"))
    end

    # left-to-right QR sweep (left-canonical tensors)
    for i in src_left:dst_left
        canonize_site!(tn, Lane(i), Bond(Lane(i), Lane(i + 1)); method=:qr)
    end

    # right-to-left QR sweep (right-canonical tensors)
    for i in src_right:-1:dst_right
        canonize_site!(tn, Lane(i), Bond(Lane(i - 1), Lane(i)); method=:qr)
    end

    tn.form = copy(dstform)

    return tn
end

# TODO optimize conversion from `MixedCanonical` to `Canonical`
# TODO `canonize!(::MixedCanonical, tn, ::Canonical)` should be optimized to start from any orthogonality center
function canonize!(::Form, ψ::AbstractMPO, ::Canonical)
    # right-to-left QR sweep, get right-canonical tensors
    canonize!(ψ, MixedCanonical(lane"1"))

    # left-to-right SVD sweep, get left-canonical tensors and singular values without reversing
    for i in 1:(nlanes(ψ) - 1)
        bond = Bond(Lane(i), Lane(i + 1))
        canonize_site!(ψ, Lane(i), bond; method=:svd, absorb=nothing)

        # extract the singular values and contract them with the next tensor
        # NOTE do not remove them, since they will be needed but TN can in be in a inconsistent state while processing
        Λᵢ = tensors(ψ; bond)

        Aᵢ₊₁ = tensors(ψ; at=Lane(i + 1))
        replace!(ψ, Aᵢ₊₁ => contract(Aᵢ₊₁, Λᵢ; dims=Symbol[]))
    end

    # tensors at i in "A" form, need to contract (Λᵢ)⁻¹ with A to get Γᵢ
    for i in 2:nlanes(ψ)
        bond = Bond(Lane(i - 1), Lane(i))
        Λᵢ = tensors(ψ; bond)
        Aᵢ = tensors(ψ; at=Lane(i))
        Λᵢ⁻¹ = Tensor(diag(pinv(Diagonal(parent(Λᵢ)); atol=1e-64)), inds(Λᵢ))
        Γᵢ = contract(Aᵢ, Λᵢ⁻¹; dims=())
        replace!(ψ, Aᵢ => Γᵢ)
    end

    ψ.form = Canonical()

    return ψ
end

function canonize!(::Canonical, ψ::AbstractMPO, ::Canonical; sweep=true)
    canonize!(ψ, MixedCanonical(lane"1"); sweep)
    canonize!(ψ, Canonical())
end

function absorb!(ψ::AbstractMPO, bond::Bond, dir::Symbol)
    targetlane = if dir === :left
        min(lanes(bond)...)
    elseif dir === :right
        max(lanes(bond)...)
    else
        throw(ArgumentError("Unknown direction for $T = :$dir"))
    end

    absorb!(ψ, bond, targetlane)
end

LinearAlgebra.norm(ψ::AbstractMPO) = norm(form(ψ), ψ)

# this is faster than mixed-canonizing
function LinearAlgebra.norm(::NonCanonical, ψ::AbstractMPO)
    tn = stack(ψ, ψ')
    sqrt(only(contract(tn)))
end

function LinearAlgebra.norm(config::MixedCanonical, tn::AbstractMPO)
    @assert config.orthog_center isa Lane "Orthogonality center must be a `Lane`"
    orthog_center = tensors(tn; at=config.orthog_center)
    return norm(orthog_center)
end

# apparently, the norm is the productory of the norms of the Λ tensors
# but since the tensor network might be not normalized, it's then the ratio against the maximum Λ norm
function LinearAlgebra.norm(::Canonical, tn::AbstractMPO)
    norms = map(bonds(tn)) do bond
        norm(tensors(tn; bond))
    end
    max_norm = maximum(norms)
    return prod(norms ./ max_norm) * max_norm
end

LinearAlgebra.normalize!(ψ::AbstractMPO; kwargs...) = normalize!(form(ψ), ψ; kwargs...)
LinearAlgebra.normalize!(ψ::AbstractMPO, at::Lane) = normalize!(form(ψ), ψ; at)
LinearAlgebra.normalize!(ψ::AbstractMPO, bond::Bond) = normalize!(form(ψ), ψ; bond)

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

# orthogonality center contains all the norm, so just normalize that tensor
function LinearAlgebra.normalize!(config::MixedCanonical, ψ::AbstractMPO)
    normalize!(tensors(ψ; at=config.orthog_center), 2)
    return ψ
end

function LinearAlgebra.normalize!(::Canonical, ψ::AbstractMPO; bond=nothing)
    if !isnothing(bond)
        # when setting `bond`, we are just normalizing one Λ tensor and its neighbor Γ tensors
        Λab = tensors(ψ; bond)
        normalize!(Λab)

        a, b = bond
        Γa, Γb = tensors(ψ; at=a), tensors(ψ; at=b)

        # ρ are Γ tensors with neighbor Λ tensors contracted => ρ = Λ Γ Λ
        # i.e. it's half reduced density matrix for the site, so it's norm is the total norm too
        # NOTE this works only if the state is correctly canonized!
        ρa, ρb = contract(Γa, Λab; dims=Symbol[]), contract(Γb, Λab; dims=Symbol[])

        # open boundary conditions
        if a != lane"1"
            Λa = tensors(ψ; bond=(Lane(id(a) - 1), a))
            ρa = contract(ρa, Λa; dims=Symbol[])
        end

        if b != Lane(nlanes(ψ))
            Λb = tensors(ψ; bond=(b, Lane(id(b) + 1)))
            ρb = contract(ρb, Λb; dims=Symbol[])
        end

        Za, Zb = norm(ρa), norm(ρb)

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
        ρ = contract(contract(Γ, Λᵢ₋₁; dims=Symbol[]), Λᵢ₊₁; dims=Symbol[])
        Z = norm(ρ)
        Γ ./= Z
    end

    # normalize the first and last Γ tensors
    Γ = tensors(ψ; at=lane"1")
    Λ = tensors(ψ; bond=(lane"1", lane"2"))
    ρ = contract(Γ, Λ; dims=Symbol[])
    Z = norm(ρ)
    Γ ./= Z

    Γ = tensors(ψ; at=Lane(nlanes(ψ)))
    Λ = tensors(ψ; bond=(Lane(nlanes(ψ) - 1), Lane(nlanes(ψ))))
    ρ = contract(Γ, Λ; dims=Symbol[])
    Z = norm(ρ)
    Γ ./= Z

    return ψ
end

# TODO improve over `evolve!` methods?

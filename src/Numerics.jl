using OMEinsum
using LinearAlgebra
using UUIDs: uuid4
using SparseArrays

# TODO test array container typevar on output
for op in [
    :+, :-, :*, :/, :\, :^, :÷, :fld, :cld, :mod, :%, :fldmod, :fld1, :mod1, :fldmod1, ://, :gcd, :lcm, :gcdx, :widemul
]
    @eval Base.$op(a::Tensor{A,0}, b::Tensor{B,0}) where {A,B} = broadcast($op, a, b)
end

function Base.literal_pow(f, a::Tensor{T,0}, ::Val{p}) where {T,p}
    return Tensor(fill(Base.literal_pow(f, only(a), Val(p))))
end

"""
    +(::Tensor, ::Tensor)

Add two tensors element-wise. The tensors must have the same indices, alghough the order of the indices can be different.
"""
function Base.:(+)(a::Tensor, b::Tensor)
    issetequal(inds(a), inds(b)) || throw(ArgumentError("indices must be equal"))
    perm = __find_index_permutation(inds(a), inds(b))
    return Tensor(parent(a) + PermutedDimsArray(parent(b), perm), inds(a))
end

"""
    -(::Tensor, ::Tensor)

Subtract two tensors element-wise. The tensors must have the same indices, alghough the order of the indices can be different.
"""
function Base.:(-)(a::Tensor, b::Tensor)
    issetequal(inds(a), inds(b)) || throw(ArgumentError("indices must be equal"))
    perm = __find_index_permutation(inds(a), inds(b))
    return Tensor(parent(a) - PermutedDimsArray(parent(b), perm), inds(a))
end

"""
    contract(a::Tensor, b::Tensor; dims=∩(inds(a), inds(b)), out=nothing)

Perform a binary tensor contraction operation.

# Keyword arguments

    - `dims`: indices to contract over. Defaults to the set intersection of the indices of `a` and `b`.
    - `out`: indices of the output tensor. Defaults to the set difference of the indices of `a` and `b`.
"""
function contract(a::Tensor, b::Tensor; dims=(∩(inds(a), inds(b))), out=nothing)
    ia = collect(inds(a))
    ib = collect(inds(b))
    i = ∩(dims, ia, ib)

    ic::Vector{Symbol} = if isnothing(out)
        setdiff(ia ∪ ib, i isa Base.AbstractVecOrTuple ? i : (i,))::Vector{Symbol}
    else
        out
    end

    data = OMEinsum.get_output_array((parent(a), parent(b)), [size(i in ia ? a : b, i) for i in ic]; fillzero=false)
    c = Tensor(data, ic)
    return contract!(c, a, b)
end

"""
    contract(a::Tensor; dims=∩(inds(a), inds(b)), out=nothing)

Perform a unary tensor contraction operation.

# Keyword arguments

    - `dims`: indices to contract over. Defaults to the repeated indices.
    - `out`: indices of the output tensor. Defaults to the unique indices.
"""
function contract(a::Tensor; dims=nonunique(inds(a)), out=nothing)
    ia = inds(a)
    i = ∩(dims, ia)

    ic::Vector{Symbol} = if isnothing(out)
        setdiff(ia, i isa Base.AbstractVecOrTuple ? i : (i,))
    else
        out
    end

    data = OMEinsum.get_output_array((parent(a),), [size(a, i) for i in ic]; fillzero=false)
    c = Tensor(data, ic)
    return contract!(c, a)
end

contract(a::Union{T,AbstractArray{T,0}}, b::Tensor{T}) where {T} = contract(Tensor(a), b)
contract(a::Tensor{T}, b::Union{T,AbstractArray{T,0}}) where {T} = contract(a, Tensor(b))
contract(a::AbstractArray{<:Any,0}, b::AbstractArray{<:Any,0}) = only(contract(Tensor(a), Tensor(b)))
contract(a::Number, b::Number) = contract(fill(a), fill(b))
contract(tensors::Tensor...; kwargs...) = reduce((x, y) -> contract(x, y; kwargs...), tensors)

"""
    contract!(c::Tensor, a::Tensor, b::Tensor)

Perform a binary tensor contraction operation between `a` and `b` and store the result in `c`.
"""
function contract!(c::Tensor, a::Tensor, b::Tensor)
    ixs = (inds(a), inds(b))
    iy = inds(c)
    xs = (parent(a), parent(b))
    y = parent(c)
    size_dict = merge!(Dict{Symbol,Int}.([inds(a) .=> size(a), inds(b) .=> size(b)])...)

    einsum!(ixs, iy, xs, y, true, false, size_dict)
    return c
end

"""
    contract!(c::Tensor, a::Tensor)

Perform a unary tensor contraction operation on `a` and store the result in `c`.
"""
function contract!(y::Tensor, x::Tensor)
    ixs = (inds(x),)
    iy = inds(y)
    size_dict = Dict{Symbol,Int}(inds(x) .=> size(x))

    einsum!(ixs, iy, (parent(x),), parent(y), true, false, size_dict)
    return y
end

"""
    *(::Tensor, ::Tensor)

Alias for [`contract`](@ref).
"""
Base.:*(a::Tensor, b::Tensor) = contract(a, b)
Base.:*(a::T, b::Number) where {T<:Tensor} = T(parent(a) * b, inds(a))
Base.:*(a::Number, b::T) where {T<:Tensor} = T(a * parent(b), inds(b))

function factorinds(tensor, left_inds, right_inds)
    isdisjoint(left_inds, right_inds) ||
        throw(ArgumentError("left ($left_inds) and right $(right_inds) indices must be disjoint"))

    left_inds, right_inds = if isempty(left_inds)
        (setdiff(inds(tensor), right_inds), right_inds)
    elseif isempty(right_inds)
        (left_inds, setdiff(inds(tensor), left_inds))
    else
        (left_inds, right_inds)
    end

    all(!isempty, (left_inds, right_inds)) || throw(ArgumentError("no right-indices left in factorization"))
    all(∈(inds(tensor)), left_inds ∪ right_inds) || throw(ArgumentError("indices must be in $(inds(tensor))"))

    return left_inds, right_inds
end

LinearAlgebra.svd(t::Tensor{<:Any,2}; kwargs...) = Base.@invoke svd(t::Tensor; left_inds=(first(inds(t)),), kwargs...)

"""
    LinearAlgebra.svd(tensor::Tensor; left_inds, right_inds, virtualind, kwargs...)

Perform SVD factorization on a tensor.

# Keyword arguments

  - `left_inds`: left indices to be used in the SVD factorization. Defaults to all indices of `t` except `right_inds`.
  - `right_inds`: right indices to be used in the SVD factorization. Defaults to all indices of `t` except `left_inds`.
  - `virtualind`: name of the virtual bond. Defaults to a random `Symbol`.
"""
function LinearAlgebra.svd(tensor::Tensor; left_inds=(), right_inds=(), virtualind=Symbol(uuid4()), kwargs...)
    left_inds, right_inds = factorinds(tensor, left_inds, right_inds)

    virtualind ∉ inds(tensor) ||
        throw(ArgumentError("new virtual bond name ($virtualind) cannot be already be present"))

    # permute array
    left_sizes = map(Base.Fix1(size, tensor), left_inds)
    right_sizes = map(Base.Fix1(size, tensor), right_inds)
    tensor = permutedims(tensor, [left_inds..., right_inds...])
    data = reshape(parent(tensor), prod(left_sizes), prod(right_sizes))

    # compute SVD
    U, s, V = svd(data; kwargs...)

    # tensorify results
    U = Tensor(reshape(U, left_sizes..., size(U, 2)), [left_inds..., virtualind])
    s = Tensor(s, [virtualind])
    Vt = Tensor(reshape(conj(V), right_sizes..., size(V, 2)), [right_inds..., virtualind])

    return U, s, Vt
end

LinearAlgebra.qr(t::Tensor{<:Any,2}; kwargs...) = Base.@invoke qr(t::Tensor; left_inds=(first(inds(t)),), kwargs...)

"""
    LinearAlgebra.qr(tensor::Tensor; left_inds, right_inds, virtualind, kwargs...)

Perform QR factorization on a tensor.

# Keyword arguments

  - `left_inds`: left indices to be used in the QR factorization. Defaults to all indices of `t` except `right_inds`.
  - `right_inds`: right indices to be used in the QR factorization. Defaults to all indices of `t` except `left_inds`.
  - `virtualind`: name of the virtual bond. Defaults to a random `Symbol`.
"""
function LinearAlgebra.qr(tensor::Tensor; left_inds=(), right_inds=(), virtualind::Symbol=Symbol(uuid4()), kwargs...)
    left_inds, right_inds = factorinds(tensor, left_inds, right_inds)

    virtualind ∉ inds(tensor) ||
        throw(ArgumentError("new virtual bond name ($virtualind) cannot be already be present"))

    # permute array
    left_sizes = map(Base.Fix1(size, tensor), left_inds)
    right_sizes = map(Base.Fix1(size, tensor), right_inds)
    tensor = permutedims(tensor, [left_inds..., right_inds...])
    data = reshape(parent(tensor), prod(left_sizes), prod(right_sizes))

    # compute QR
    F = qr(data; kwargs...)
    Q, R = Matrix(F.Q), Matrix(F.R)

    # tensorify results
    Q = Tensor(reshape(Q, left_sizes..., size(Q, 2)), [left_inds..., virtualind])
    R = Tensor(reshape(R, size(R, 1), right_sizes...), [virtualind, right_inds...])

    return Q, R
end

LinearAlgebra.lu(t::Tensor{<:Any,2}; kwargs...) = Base.@invoke lu(t::Tensor; left_inds=(first(inds(t)),), kwargs...)

"""
    LinearAlgebra.lu(tensor::Tensor; left_inds, right_inds, virtualind, kwargs...)

Perform LU factorization on a tensor.

# Keyword arguments

  - `left_inds`: left indices to be used in the LU factorization. Defaults to all indices of `t` except `right_inds`.
  - `right_inds`: right indices to be used in the LU factorization. Defaults to all indices of `t` except `left_inds`.
  - `virtualind`: name of the virtual bond. Defaults to a random `Symbol`.
"""
function LinearAlgebra.lu(
    tensor::Tensor; left_inds=(), right_inds=(), virtualind=[Symbol(uuid4()), Symbol(uuid4())], kwargs...
)
    left_inds, right_inds = factorinds(tensor, left_inds, right_inds)

    i_pl, i_lu = virtualind
    i_pl ∉ inds(tensor) || throw(ArgumentError("new virtual bond name ($i_pl) cannot be already be present"))
    i_lu ∉ inds(tensor) || throw(ArgumentError("new virtual bond name ($i_lu) cannot be already be present"))

    # permute array
    left_sizes = map(Base.Fix1(size, tensor), left_inds)
    right_sizes = map(Base.Fix1(size, tensor), right_inds)
    tensor = permutedims(tensor, [left_inds..., right_inds...])
    data = reshape(parent(tensor), prod(left_sizes), prod(right_sizes))

    # compute LU
    info = lu(data; kwargs...)
    L = info.L
    U = info.U

    permutator = info.p
    P = sparse(permutator, 1:length(permutator), fill(true, length(permutator)))

    L = Tensor(L, [i_pl, i_lu])
    U = Tensor(reshape(U, size(U, 1), right_sizes...), [i_lu, right_inds...])
    P = Tensor(reshape(P, left_sizes..., size(L, 1)), [left_inds..., i_pl])

    return L, U, P
end

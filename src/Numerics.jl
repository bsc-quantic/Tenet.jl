using OMEinsum
using LinearAlgebra
using UUIDs: uuid4

# TODO test array container typevar on output
for op in [
    :+,
    :-,
    :*,
    :/,
    :\,
    :^,
    :÷,
    :fld,
    :cld,
    :mod,
    :%,
    :fldmod,
    :fld1,
    :mod1,
    :fldmod1,
    ://,
    :gcd,
    :lcm,
    :gcdx,
    :widemul,
]
    @eval Base.$op(a::Tensor{A,0}, b::Tensor{B,0}) where {A,B} = broadcast($op, a, b)
end

"""
    contract(a::Tensor[, b::Tensor, dims=nonunique([labels(a)..., labels(b)...])])

Perform tensor contraction operation.
"""
function contract(a::Tensor, b::Tensor; dims = (∩(labels(a), labels(b))))
    ia = labels(a)
    ib = labels(b)
    i = ∩(dims, ia, ib)

    ic = tuple(setdiff(ia ∪ ib, i isa Base.AbstractVecOrTuple ? i : (i,))...)

    data = EinCode((String.(ia), String.(ib)), String.(ic))(parent(a), parent(b))

    # TODO merge metadata?
    return Tensor(data, ic)
end

function contract(a::Tensor; dims = nonunique(labels(a)))
    ia = labels(a)
    i = ∩(dims, ia)

    ic = tuple(setdiff(ia, i isa Base.AbstractVecOrTuple ? i : (i,))...)

    data = EinCode((String.(ia),), String.(ic))(parent(a))

    # TODO merge metadata
    return Tensor(data, ic)
end

contract(a::Union{T,AbstractArray{T,0}}, b::Tensor{T}) where {T} = contract(Tensor(a), b)
contract(a::Tensor{T}, b::Union{T,AbstractArray{T,0}}) where {T} = contract(a, Tensor(b))
contract(a::AbstractArray{<:Any,0}, b::AbstractArray{<:Any,0}) = contract(Tensor(a), Tensor(b)) |> only
contract(a::Number, b::Number) = contract(fill(a), fill(b))
contract(tensors::Tensor...; kwargs...) = reduce((x, y) -> contract(x, y; kwargs...), tensors)

"""
    *(::Tensor, ::Tensor)

Alias for [`contract`](@ref).
"""
Base.:*(a::Tensor, b::Tensor) = contract(a, b)
Base.:*(a::Tensor, b) = contract(a, b)
Base.:*(a, b::Tensor) = contract(a, b)

LinearAlgebra.svd(t::Tensor{<:Any,2}; kwargs...) =
    Base.@invoke svd(t::Tensor; left_inds = (first(labels(t)),), kwargs...)

function LinearAlgebra.svd(t::Tensor; left_inds, kwargs...)
    if isempty(left_inds)
        throw(ErrorException("no left-indices in SVD factorization"))
    elseif any(∉(labels(t)), left_inds)
        # TODO better error exception and checks
        throw(ErrorException("all left-indices must be in $(labels(t))"))
    end

    right_inds = setdiff(labels(t), left_inds)
    if isempty(right_inds)
        # TODO better error exception and checks
        throw(ErrorException("no right-indices in SVD factorization"))
    end

    # permute array
    tensor = permutedims(t, (left_inds..., right_inds...))
    data = reshape(parent(tensor), prod(i -> size(t, i), left_inds), prod(i -> size(t, i), right_inds))

    # compute SVD
    U, s, V = svd(data; kwargs...)

    # tensorify results
    U = reshape(U, ([size(t, ind) for ind in left_inds]..., size(U, 2)))
    s = Diagonal(s)
    Vt = reshape(V', (size(V', 1), [size(t, ind) for ind in right_inds]...))

    vlind = Symbol(uuid4())
    vrind = Symbol(uuid4())

    U = Tensor(U, (left_inds..., vlind))
    s = Tensor(s, (vlind, vrind))
    Vt = Tensor(Vt, (vrind, right_inds...))

    return U, s, Vt
end

LinearAlgebra.qr(t::Tensor{<:Any,2}; kwargs...) = Base.@invoke qr(t::Tensor; left_inds = (first(labels(t)),), kwargs...)

function LinearAlgebra.qr(t::Tensor; left_inds = (), right_inds = (), virtualind::Symbol = Symbol(uuid4()), kwargs...)
    isdisjoint(left_inds, right_inds) ||
        throw(ArgumentError("left ($left_inds) and right $(right_inds) indices must be disjoint"))

    left_inds, right_inds =
        isempty(left_inds) ? (setdiff(labels(t), right_inds), right_inds) :
        isempty(right_inds) ? (left_inds, setdiff(labels(t), left_inds)) :
        throw(ArgumentError("cannot set both left and right indices"))

    all(!isempty, (left_inds, right_inds)) || throw(ArgumentError("no right-indices left in QR factorization"))
    all(∈(labels(t)), left_inds ∪ right_inds) || throw(ArgumentError("indices must be in $(labels(t))"))

    virtualind ∉ labels(t) || throw(ArgumentError("new virtual bond name ($virtualind) cannot be already be present"))

    # permute array
    tensor = permutedims(t, (left_inds..., right_inds...))
    data = reshape(parent(tensor), prod(i -> size(t, i), left_inds), prod(i -> size(t, i), right_inds))

    # compute QR
    Q, R = qr(data; kwargs...)

    # tensorify results
    Q = reshape(Q, ([size(t, ind) for ind in left_inds]..., size(Q, 2)))
    R = reshape(R, (size(R, 1), [size(t, ind) for ind in right_inds]...))

    Q = Tensor(Q, (left_inds..., virtualind))
    R = Tensor(R, (virtualind, right_inds...))

    return Q, R
end

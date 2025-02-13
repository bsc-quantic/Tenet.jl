using OMEinsum: OMEinsum

# default methods
function contract(@nospecialize(::T), @nospecialize(::Tensor); kwargs...) where {T<:AbstractBackend}
    error("Backend `$T` not implemented")
end

function contract(
    @nospecialize(::T), @nospecialize(::Tensor), @nospecialize(::Tensor); kwargs...
) where {T<:AbstractBackend}
    error("Backend `$T` not implemented")
end

contract(a::Union{T,AbstractArray{T,0}}, b::Tensor{T}) where {T} = contract(Tensor(a), b)
contract(a::Tensor{T}, b::Union{T,AbstractArray{T,0}}) where {T} = contract(a, Tensor(b))
contract(a::AbstractArray{<:Any,0}, b::AbstractArray{<:Any,0}) = only(contract(Tensor(a), Tensor(b)))
contract(a::Number, b::Number) = contract(fill(a), fill(b))

contract(tensors::Tensor...; kwargs...) = contract(default_backend[], tensors...; kwargs...)

function contract(b::AbstractBackend, tensors::Tensor...; kwargs...)
    if length(tensors) > 8
        @info "Contracting $(length(tensors)) tensors without searching for the contraction path may be slow. Using `contract(TensorNetwork(tensors))` instead."
    end
    reduce((x, y) -> contract(b, x, y; kwargs...), tensors)
end

# TODO check out that this is not incurring in a big performance penalty due to dynamic dispatch
# NOTE if overhead, we can use a generated function here that gets invalidated whenever we change the default backend
"""
    contract(a::Tensor, b::Tensor; dims=∩(inds(a), inds(b)), out=nothing)

Perform a binary tensor contraction operation.

# Keyword arguments

    - `dims`: indices to contract over. Defaults to the set intersection of the indices of `a` and `b`.
    - `out`: indices of the output tensor. Defaults to the set difference of the indices of `a` and `b`.

!!! todo

    We are in the process of making [`contract`](@ref) multi-backend; i.e. let the user choose between different einsum libraries as the engine powering [`contract`](@ref).
    Currently, we use [OMEinsum.jl](@ref), but it has proven to be slow when used dynamically like we do.
"""
contract(a::Tensor, b::Tensor; kwargs...) = contract(default_backend[], a, b; kwargs...)

"""
    contract(a::Tensor; dims=∩(inds(a), inds(b)), out=nothing)

Perform a unary tensor contraction operation.

# Keyword arguments

    - `dims`: indices to contract over. Defaults to the repeated indices.
    - `out`: indices of the output tensor. Defaults to the unique indices.
"""
contract(a::Tensor; kwargs...) = contract(default_backend[], a; kwargs...)

"""
    contract!(c::Tensor, a::Tensor, b::Tensor)

Perform a binary tensor contraction operation between `a` and `b` and store the result in `c`.
"""
contract!(c::Tensor, a::Tensor, b::Tensor) = contract!(default_backend[], c, a, b)

"""
    contract!(c::Tensor, a::Tensor)

Perform a unary tensor contraction operation on `a` and store the result in `c`.
"""
contract!(y::Tensor, x::Tensor) = contract!(default_backend[], y, x)

# OMEinsum backend
function contract(::OMEinsumBackend, a::Tensor, b::Tensor; kwargs...)
    c = allocate_result(contract, a, b; kwargs...)
    return contract!(OMEinsumBackend(), c, a, b)
end

function allocate_result(
    ::typeof(contract), a::Tensor, b::Tensor; fillzero=false, dims=(∩(inds(a), inds(b))), out=nothing
)
    ia = collect(inds(a))
    ib = collect(inds(b))
    i = ∩(dims, ia, ib)

    ic = if isnothing(out)
        Tuple(setdiff(ia ∪ ib, i isa Base.AbstractVecOrTuple ? i : (i,)))
    else
        out
    end

    data = OMEinsum.get_output_array((parent(a), parent(b)), [size(i in ia ? a : b, i) for i in ic]; fillzero)
    return Tensor(data, ic)
end

function contract(::OMEinsumBackend, a::Tensor; kwargs...)
    c = allocate_result(contract, a; kwargs...)
    return contract!(OMEinsumBackend(), c, a)
end

function allocate_result(::typeof(contract), a::Tensor; fillzero=false, dims=nonunique(inds(a)), out=nothing)
    ia = inds(a)
    i = ∩(dims, ia)

    ic::Vector{Symbol} = if isnothing(out)
        setdiff(ia, i isa Base.AbstractVecOrTuple ? i : (i,))
    else
        out
    end

    data = OMEinsum.get_output_array((parent(a),), [size(a, i) for i in ic]; fillzero)
    return Tensor(data, ic)
end

function contract!(::OMEinsumBackend, c::Tensor, a::Tensor, b::Tensor)
    ixs = (inds(a), inds(b))
    iy = inds(c)
    xs = (parent(a), parent(b))
    y = parent(c)
    size_dict = merge!(Dict{Symbol,Int}.([inds(a) .=> size(a), inds(b) .=> size(b)])...)

    OMEinsum.einsum!(ixs, iy, xs, y, true, false, size_dict)
    return c
end

"""
    contract!(c::Tensor, a::Tensor)

Perform a unary tensor contraction operation on `a` and store the result in `c`.
"""
function contract!(::OMEinsumBackend, y::Tensor, x::Tensor)
    ixs = (inds(x),)
    iy = inds(y)
    size_dict = Dict{Symbol,Int}(inds(x) .=> size(x))

    OMEinsum.einsum!(ixs, iy, (parent(x),), parent(y), true, false, size_dict)
    return y
end

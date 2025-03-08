using Base: @propagate_inbounds
using Base.Broadcast: Broadcasted, ArrayStyle
using LinearAlgebra

"""
    Tensor{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}

An array-like object with named dimensions.
"""
struct Tensor{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
    inds::Vector{Symbol}

    function Tensor{T,N,A}(data::A, inds) where {T,N,A<:AbstractArray{T,N}}
        inds = collect(inds)
        length(inds) == N ||
            throw(ArgumentError("ndims(data) [$(ndims(data))] must be equal to length(inds) [$(length(inds))]"))
        all(i -> allequal(Iterators.map(dim -> size(data, dim), findall(==(i), inds))), nonunique(inds)) ||
            throw(DimensionMismatch("nonuniform size of repeated indices"))

        return new{T,N,A}(data, inds)
    end
end

"""
    Tensor(data::AbstractArray{T,N}, inds::AbstractVector{Symbol})
    Tensor(data::AbstractArray{T,N}, inds::NTuple{N,Symbol}) where {T,N}
    Tensor(data::AbstractArray{T,0}) where {T}
    Tensor(data::Number)

Construct a tensor with the given data and indices.
"""
Tensor(data::A, inds::AbstractVector{Symbol}) where {T,N,A<:AbstractArray{T,N}} = Tensor{T,N,A}(data, inds)
Tensor(data::A, inds::NTuple{N,Symbol}) where {T,N,A<:AbstractArray{T,N}} = Tensor{T,N,A}(data, collect(inds))
Tensor(data::AbstractArray{T,0}) where {T} = Tensor(data, Symbol[])
Tensor(data::Number) = Tensor(fill(data))

"""
    inds(tensor::Tensor)

Return the indices of the tensor in the order of the dimensions.
"""
inds(t::Tensor) = Tuple(t.inds)

# WARN internal use only because it can mutate `Tensor`
vinds(t::Tensor) = t.inds

function Base.copy(t::Tensor{T,N,<:SubArray{T,N}}) where {T,N}
    data = copy(t.data)
    return Tensor(data, vinds(t))
end

"""
    Base.similar(::Tensor{T,N}[, S::Type, dims::Base.Dims{N}; inds])

Return a uninitialize tensor of the same size, eltype and [`inds`](@ref) as `tensor`. If `S` is provided, the eltype of the tensor will be `S`. If `dims` is provided, the size of the tensor will be `dims`.
"""
Base.similar(t::Tensor; inds=inds(t)) = Tensor(similar(parent(t)), inds)
Base.similar(t::Tensor, S::Type; inds=inds(t)) = Tensor(similar(parent(t), S), inds)
function Base.similar(t::Tensor{T,N}, S::Type, dims::Base.Dims{N}; inds=vinds(t)) where {T,N}
    return Tensor(similar(parent(t), S, dims), inds)
end
function Base.similar(t::Tensor, ::Type, dims::Base.Dims{N}; kwargs...) where {N}
    throw(DimensionMismatch("`dims` needs to be of length $(ndims(t))"))
end
Base.similar(t::Tensor{T,N}, dims::Base.Dims{N}; inds=inds(t)) where {T,N} = Tensor(similar(parent(t), dims), inds)
function Base.similar(t::Tensor, dims::Base.Dims{N}; kwargs...) where {N}
    throw(DimensionMismatch("`dims` needs to be of length $(ndims(t))"))
end

"""
    Base.zero(tensor::Tensor)

Return a tensor of the same size, eltype and [`inds`](@ref) as `tensor` but filled with zeros.
"""
Base.zero(t::Tensor) = Tensor(zero(parent(t)), vinds(t))

function __find_index_permutation(a, b)
    inds_b = collect(Union{Missing,Symbol}, b)

    return collect(
        Iterators.map(a) do label
            i = findfirst(isequal(label), inds_b)

            # mark element as used
            inds_b[i] = missing

            i
        end,
    )
end

Base.:(==)(a::AbstractArray, b::Tensor) = isequal(b, a)
Base.:(==)(a::Tensor, b::AbstractArray) = isequal(a, b)
Base.:(==)(a::Tensor, b::Tensor) = isequal(a, b)
Base.isequal(a::AbstractArray, b::Tensor) = false
Base.isequal(a::Tensor, b::AbstractArray) = false
function Base.isequal(a::Tensor, b::Tensor)
    issetequal(inds(a), inds(b)) || return false
    perm = __find_index_permutation(inds(a), inds(b))
    return all(eachindex(IndexCartesian(), a)) do i
        j = CartesianIndex(Tuple(permute!(collect(Tuple(i)), invperm(perm))))
        isequal(a[i], b[j])
    end
end

Base.isequal(a::Tensor{A,0}, b::Tensor{B,0}) where {A,B} = isequal(only(a), only(b))

Base.isapprox(a::AbstractArray, b::Tensor) = false
Base.isapprox(a::Tensor, b::AbstractArray) = false
function Base.isapprox(a::Tensor, b::Tensor; kwargs...)
    issetequal(vinds(a), vinds(b)) || return false
    perm = __find_index_permutation(vinds(a), vinds(b))
    return all(eachindex(IndexCartesian(), a)) do i
        j = CartesianIndex(Tuple(permute!(collect(Tuple(i)), invperm(perm))))
        isapprox(a[i], b[j]; kwargs...)
    end
end

Base.isapprox(a::Tensor{T,0}, b::T; kwargs...) where {T} = isapprox(only(a), b; kwargs...)
Base.isapprox(a::T, b::Tensor{T,0}; kwargs...) where {T} = isapprox(b, a; kwargs...)
Base.isapprox(a::Tensor{A,0}, b::Tensor{B,0}; kwargs...) where {A,B} = isapprox(only(a), only(b); kwargs...)

# NOTE: `replace` does not currenly support cyclic replacements
"""
    Base.replace(::Tensor, old_new::Pair{Symbol,Symbol}...)

Replace the indices of the tensor according to the given pairs of old and new indices.

!!! warning

    This method does not support cyclic replacements.
"""
Base.replace(t::Tensor, old_new::Pair{Symbol,Symbol}...) = Tensor(parent(t), replace(inds(t), old_new...))
Base.replace!(t::Tensor, old_new::Pair...) = throw(MethodError(Base.replace!, (t, old_new...)))

"""
    Base.parent(::Tensor)

Return the underlying array of the tensor.
"""
Base.parent(t::Tensor) = t.data
parenttype(::Type{Tensor{T,N,A}}) where {T,N,A} = A
parenttype(::Type{Tensor{T,N}}) where {T,N} = AbstractArray{T,N}
parenttype(::Type{Tensor{T}}) where {T} = AbstractArray{T}
parenttype(::Type{Tensor}) = AbstractArray
parenttype(::T) where {T<:Tensor} = parenttype(T)

"""
    dim(tensor::Tensor, i::Symbol)

Return the location of the dimension of `tensor` corresponding to the given index `i`.
"""
dim(::Tensor, i::Number) = i
dim(t::Tensor, i::Symbol) = findfirst(==(i), vinds(t))

# Iteration interface
Base.IteratorSize(T::Type{Tensor}) = Iterators.IteratorSize(parenttype(T))
Base.IteratorEltype(T::Type{Tensor}) = Iterators.IteratorEltype(parenttype(T))

Base.isdone(t::Tensor) = Base.isdone(parent(t))
Base.isdone(t::Tensor, state) = Base.isdone(parent(t), state)

# Indexing interface
Base.IndexStyle(T::Type{<:Tensor}) = IndexStyle(parenttype(T))

"""
    Base.getindex(::Tensor, i...)
    Base.getindex(::Tensor; i...)
    (::Tensor)[index=i...]

Return the element of the tensor at the given indices. If kwargs are provided, then it is equivalent to calling [`view`](@ref).
"""
@propagate_inbounds Base.getindex(t::Tensor, i...) = getindex(parent(t), i...)
@propagate_inbounds function Base.getindex(t::Tensor; i...)
    length(i) == 0 && return (getindex ∘ parent)(t)
    return getindex(t, [get(i, label, Colon()) for label in inds(t)]...)
end

"""
    Base.setindex!(t::Tensor, v, i...)
    Base.setindex(::Tensor; i...)
    (::Tensor)[index=i...]

Set the element of the tensor at the given indices to `v`. If kwargs are provided, then it is equivalent to calling `.=` on [`view`](@ref).
"""
@propagate_inbounds Base.setindex!(t::Tensor, v, i...) = setindex!(parent(t), v, i...)
@propagate_inbounds function Base.setindex!(t::Tensor, v; i...)
    length(i) == 0 && return setindex!(parent(t), v)
    return setindex!(t, v, [get(i, label, Colon()) for label in inds(t)]...)
end

Base.firstindex(t::Tensor) = firstindex(parent(t))
Base.lastindex(t::Tensor) = lastindex(parent(t))

# AbstractArray interface
"""
    Base.size(::Tensor[, i::Symbol])

Return the size of the underlying array. If the dimension `i` (specified by `Symbol` or `Integer`) is specified, then the size of the corresponding dimension is returned.
"""
Base.size(t::Tensor) = size(parent(t))
Base.size(t::Tensor, i) = size(parent(t), dim(t, i))

"""
    Base.length(::Tensor)

Return the length of the underlying array.
"""
Base.length(t::Tensor) = length(parent(t))

Base.axes(t::Tensor) = axes(parent(t))
Base.axes(t::Tensor, d) = axes(parent(t), dim(t, d))

# StridedArrays interface
Base.strides(t::Tensor) = strides(parent(t))
Base.stride(t::Tensor, i::Symbol) = stride(parent(t), dim(t, i))

Base.unsafe_convert(::Type{Ptr{T}}, t::Tensor{T}) where {T} = Base.unsafe_convert(Ptr{T}, parent(t))

Base.elsize(T::Type{<:Tensor}) = Base.elsize(parenttype(T))

# Broadcasting
Base.BroadcastStyle(::Type{T}) where {T<:Tensor} = ArrayStyle{T}()

function Base.similar(bc::Broadcasted{ArrayStyle{Tensor{T,N,A}}}, ::Type{ElType}) where {T,N,A,ElType}
    # NOTE already checked if dimension mismatch
    # TODO throw on label mismatch?
    tensor = first(arg for arg in bc.args if arg isa Tensor{T,N,A})
    return similar(tensor, ElType)
end

"""
    Base.selectdim(tensor::Tensor, dim::Symbol, i)
    Base.selectdim(tensor::Tensor, dim::Integer, i)

Return a view of the tensor where the index for dimension `dim` equals `i`.

!!! note

    This method doesn't return a `SubArray`, but a `Tensor` wrapping a `SubArray`.

See also: [`selectdim`](@ref)
"""
Base.selectdim(t::Tensor, d::Integer, i) = Tensor(selectdim(parent(t), d, i), vinds(t))
function Base.selectdim(t::Tensor, d::Integer, i::Integer)
    data = selectdim(parent(t), d, i)
    indices = [label for (i, label) in enumerate(vinds(t)) if i != d]
    return Tensor(data, indices)
end

Base.selectdim(t::Tensor, d::Symbol, i) = selectdim(t, dim(t, d), i)

"""
    Base.permutedims(tensor::Tensor, perm)

Permute the dimensions of `tensor` according to the given permutation `perm`. The [`inds`](@ref) will be permuted accordingly.
"""
Base.permutedims(t::Tensor, perm) = Tensor(permutedims(parent(t), perm), getindex.((inds(t),), perm))
Base.permutedims!(dest::Tensor, src::Tensor, perm) = permutedims!(parent(dest), parent(src), perm)

function Base.permutedims(t::Tensor{T}, perm::Base.AbstractVecOrTuple{Symbol}) where {T}
    perm = map(i -> findfirst(==(i), inds(t)), perm)
    return permutedims(t, perm)
end

"""
    Base.dropdims(tensor::Tensor; dims)

Return a tensor where the dimensions specified by `dims` are removed. `size(tensor, dim) == 1` for each dimension in `dims`.
"""
function Base.dropdims(t::Tensor; dims=tuple(findall(==(1), size(t))...))
    return Tensor(dropdims(parent(t); dims), inds(t)[setdiff(1:ndims(t), dims)])
end

"""
    Base.view(tensor::Tensor, i...)
    Base.view(tensor::Tensor, inds::Pair{Symbol,<:Any}...)

Return a view of the tensor with the given indices. If a `Pair` is given, the index is replaced by the value of the pair.

!!! note

    This method doesn't return a `SubArray`, but a `Tensor` wrapping a `SubArray`.
"""
function Base.view(t::Tensor, i...)
    return Tensor(view(parent(t), i...), [label for (label, j) in zip(inds(t), i) if !(j isa Integer)])
end

function Base.view(t::Tensor, inds::Pair{Symbol,<:Any}...)
    indices = map(Tenet.inds(t)) do ind
        i = findfirst(x -> x == ind, first.(inds))
        !isnothing(i) ? inds[i].second : Colon()
    end

    let data = view(parent(t), indices...),
        inds = [label for (index, label) in zip(indices, Tenet.inds(t)) if !(index isa Integer)]

        Tensor(data, inds)
    end
end

# NOTE: `conj` is automatically managed because `Tensor` inherits from `AbstractArray`,
# but there is a bug when calling `conj` on `Tensor{T,0}` which makes it return a `Tensor{Tensor{Complex, 0}, 0}`
"""
    Base.conj(::Tensor)

Return the conjugate of the tensor.
"""
Base.conj(x::Tensor{<:Complex,0}) = Tensor(conj(parent(x)), Symbol[])

"""
    Base.adjoint(::Tensor)

Return the adjoint of the tensor.

!!! note

    This method doesn't transpose the array. It is equivalent to [`conj`](@ref).
"""
Base.adjoint(t::Tensor) = conj(t)

# NOTE: Maybe use transpose for lazy transposition ?
Base.transpose(t::Tensor{T,1,A}) where {T,A<:AbstractArray{T,1}} = copy(t)
Base.transpose(t::Tensor{T,2,A}) where {T,A<:AbstractArray{T,2}} = Tensor(transpose(parent(t)), reverse(inds(t)))

"""
    expand(tensor::Tensor; label[, axis=1, size=1, method=:zeros])

Expand the tensor by adding a new dimension `label` with the given `size` at the specified `axis`.
Currently the supported methods are `:zeros` and `:repeat`.
"""
function expand(tensor::Tensor; label, axis=1, size=1, method=:zeros)
    array = parent(tensor)
    data = if size == 1
        reshape(array, Base.size(array)[1:(axis - 1)]..., 1, Base.size(array)[axis:end]...)
    elseif method === :zeros
        __expand_zeros(array, axis, size)
    elseif method === :repeat
        __expand_repeat(array, axis, size)
    else
        # method === :identity ? __expand_identity(array, axis, size) :
        throw(ArgumentError("method \"$method\" is not valid"))
    end

    inds = (Tenet.inds(tensor)[1:(axis - 1)]..., label, Tenet.inds(tensor)[axis:end]...)

    return Tensor(data, inds)
end

function __expand_zeros(array, axis, size)
    new = zeros(eltype(array), Base.size(array)[1:(axis - 1)]..., size, Base.size(array)[axis:end]...)

    view = selectdim(new, axis, 1)
    copy!(view, array)

    return new
end

function __expand_repeat(array, axis, size)
    return repeat(
        reshape(array, Base.size(array)[1:(axis - 1)]..., 1, Base.size(array)[axis:end]...);
        outer=(fill(1, axis - 1)..., size, fill(1, ndims(array) - axis + 1)...),
    )
end

LinearAlgebra.opnorm(x::Tensor, p::Real) = opnorm(parent(x), p)

# TODO choose a new index name? currently choosing the first index of `parinds`
"""
    fuse(tensor, parinds; ind=first(parinds))

Fuses `parinds`, leaves them on the right-side internally permuted with `permutator` and names it as `ind`.
"""
function fuse(tensor::Tensor, parinds; ind=first(parinds))
    @assert allunique(inds(tensor))
    @assert parinds ⊆ inds(tensor)

    locs = findall(∈(parinds), inds(tensor))
    perm = filter(∉(locs), 1:ndims(tensor))
    append!(perm, map(i -> findfirst(==(i), inds(tensor)), parinds))

    data = perm == 1:ndims(tensor) ? parent(tensor) : permutedims(parent(tensor), perm)
    data = reshape(data, (size(data)[1:(ndims(data) - length(parinds))]..., :))

    newinds = (filter(∉(parinds), inds(tensor))..., ind)
    return Tensor(data, newinds)
end
